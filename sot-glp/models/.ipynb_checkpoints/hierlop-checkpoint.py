from typing import Type, Any, Dict, Optional, List, Tuple

import math
import numpy as np

import clip
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys

from clip import load as load_clip

import gallop.lib as lib
import gallop.vlprompt.tools as vlp_tools
from gallop.vlprompt.prompted_transformers import PromptedTransformer
from gallop.vlprompt.clip_local import ModifiedResNet, VisionTransformer, CLIP

NoneType = Type[None]
KwargType = Dict[str, Any]
CLIP_NAME = {"clip_vit_b32": "ViT-B/32", "clip_vit_b16": "ViT-B/16", "clip_resnet50": "RN50", "clip_resnet101": "RN101"}


def global_weighter(global_logits,local_logits):
    #global_min, _ = global_logits.min(dim=1, keepdim=True)
    #global_max, _ = global_logits.max(dim=1, keepdim=True)
    #global_weights = (global_logits - global_min) / (global_max - global_min + 1e-8)
    global_weights = global_logits.mean(dim=-1)
    return global_weights.unsqueeze(1).unsqueeze(-1).repeat(1,local_logits.shape[1],1,local_logits.shape[-1])

def local_weighter(local_logit):
    local_min, _ = local_logit.min(dim=2, keepdim=True)
    local_max, _ = local_logit.max(dim=2, keepdim=True)
    # Local Weight dimension is B,Token_Size,Num_Classes
    local_weights = (local_logit - local_min) / (local_max - local_min + 1e-8)
    return local_weights

class Linear(nn.Module):
    def __init__(self, in_dim: int, identity_init: bool = True) -> NoneType:
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        if identity_init:
            nn.init.zeros_(self.linear.weight)
            self.linear.weight.data += torch.eye(in_dim)
        else:
            nn.init.normal_(self.linear.weight, std=in_dim**-0.5)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class HierLop(CLIP):
    TRAINABLE_PARAMS: List[str] = []

    def __init__(
        self,
        clip_name: str,
        use_local_features: bool = True,
        checkpointing_segments: int = 8,
        template: str = "A photo of a {}",
        learn_local_proj: bool = True,
        learn_local_prompts: bool = True,
        learn_global_prompt: bool = True,
        class_names: List[str] = None,
        n_global_prompts: int = 1,
        n_local_prompts: int = 1,
        prompts_batch_size: int = math.inf,
        ood_method: str = "GL-MCM",
        ood_temp_scale: float = 1.0,
        topk: List[int] = [5, 10, 15, 20],
        parallel_text_encoder: bool = False,
        parallel_vision_encoder: bool = False,
        init_method = "random",
    ) -> NoneType:
        self.model_name = "gallop_" + clip_name[5:]
        clip_model, _ = load_clip(CLIP_NAME[clip_name], device="cuda")

        clip_state_dict = clip_model.state_dict()
        clip_kwargs = lib.get_clip_hyperparams(clip_state_dict)
        clip_kwargs["return_local_features"] = use_local_features

        super().__init__(**clip_kwargs)
        self.clip_name = clip_name
        self.use_local_features = use_local_features
        self.learn_local_proj = learn_local_proj
        self.template = template[:-1] if template[-1] == "." else template
        self.learn_local_prompts = learn_local_prompts
        self.learn_global_prompt = learn_global_prompt
        self.learn_visual_prompt = False
        self.class_names = class_names
        self.n_global_prompts = n_global_prompts
        self.n_local_prompts = n_local_prompts
        self.n_visual_prompts = 2
        self.prompts_batch_size = min(prompts_batch_size, self.n_global_prompts)
        self.ood_method = ood_method
        self.ood_temp_scale = ood_temp_scale
        self.topk = topk
        

        self.parallel_text_encoder = parallel_text_encoder
        self.parallel_vision_encoder = parallel_vision_encoder
        self.init_method = init_method

        if isinstance(clip_kwargs["vision_layers"], (tuple, list)):
            self.visual = ModifiedResNet(
                layers=clip_kwargs["vision_layers"],
                output_dim=clip_kwargs["embed_dim"],
                heads=clip_kwargs["vision_width"] * 32 // 64,
                input_resolution=clip_kwargs["image_resolution"],
                width=clip_kwargs["vision_width"]
            )
            vision_dim = clip_kwargs["embed_dim"]
        else:
            self.visual = VisionTransformer(
                input_resolution=clip_kwargs["image_resolution"],
                patch_size=clip_kwargs["vision_patch_size"],
                width=clip_kwargs["vision_width"],
                layers=clip_kwargs["vision_layers"],
                heads=clip_kwargs["vision_width"] // 64,
                output_dim=clip_kwargs["embed_dim"]
            )
            vision_dim = clip_kwargs["vision_width"]

        self.transformer = PromptedTransformer(
            width=clip_kwargs["transformer_width"],
            layers=clip_kwargs["transformer_layers"],
            heads=clip_kwargs["transformer_heads"],
            attn_mask=self.build_attention_mask(),
            segments=checkpointing_segments,
        )

        self.local_proj = Linear(vision_dim)

        if self.learn_local_proj:
            self.TRAINABLE_PARAMS.append("local_proj")


        self.use_hierarchycal_prompts = False
        self.num_classes = len(self.class_names)
       
        if self.learn_global_prompt or self.learn_local_prompts or self.n_global_prompts > 1 or self.n_local_prompts > 1:
            template = self.template.replace("{}", " ").replace("_", " ").strip()
            tokenized_template = clip.tokenize(template)
            self.template_init_tokens = int(tokenized_template.argmax(dim=-1)) - 1
            self.n_token_context = self.template_init_tokens

            if self.learn_global_prompt or self.n_global_prompts > 1:
                if self.learn_global_prompt:
                    self.TRAINABLE_PARAMS.append("global_prompt")
                self.global_prompt = nn.Parameter(
                    torch.empty(self.n_global_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )

            if self.learn_local_prompts or self.n_local_prompts > 1:
                if self.learn_local_prompts:
                    self.TRAINABLE_PARAMS.append("local_prompt")
                self.local_prompts = nn.Parameter(
                    torch.empty(self.num_classes,self.n_local_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                )
                #self.local_prompts = nn.Parameter(
                #    torch.empty(self.n_local_prompts, self.n_token_context, clip_kwargs["transformer_width"]),
                #)

            self.visual.learn_visual_prompt = self.learn_visual_prompt
            if self.learn_visual_prompt or self.n_visual_prompts > 1:
                if self.learn_local_prompts:
                    self.TRAINABLE_PARAMS.append("visual_prompt")
                scale = 768 ** -0.5
                
                #self.visual_prompts = nn.Parameter(
                #    scale * torch.randn(self.n_visual_prompts, 768),
                #)
                #self.visual.visual_prompts = self.visual_prompts
                #self.visual.n_visual_prompts = self.n_visual_prompts

                self.TRAINABLE_PARAMS.append("B")
                self.TRAINABLE_PARAMS.append("A")
                #self.B = nn.Parameter(torch.randn(3, 224, 16))
                #self.A = nn.Parameter(torch.randn(3, 16, 224))
                self.B = nn.Parameter(torch.zeros(3, 224, 16))
                self.A = nn.Parameter(torch.empty(3, 16, 224))  # Standard normal distribution
                torch.nn.init.normal_(self.A, mean=0.0, std=0.01)
                
        self.initialize_parameters()

        key_issue_clip = self.load_state_dict(clip_state_dict, strict=False)
        if len(key_issue_clip.missing_keys) > 0:
            lib.LOGGER.warning(f"Missing keys in CLIP: {key_issue_clip.missing_keys}")

        self.transformer = self.transformer if not self.parallel_text_encoder else vlp_tools.DataParallel(self.transformer)


        
        
        #self.visual = self.visual if not self.parallel_vision_encoder else vlp_tools.DataParallel(self.visual)
        self.vision_layer_num = clip_kwargs["vision_layers"]
        #self.num_of_local_hierarchy = 4
        self.last_attn = nn.MultiheadAttention(768, 12)
        #self.last_attn_mask = self.build_attention_mask()
        self.multi_head_attention_layer = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            batch_first=True  # Crucial for (batch, seq, feature) input format
        )
        d_model = 512
        p_drop = 0
        self.ln1_begin = nn.LayerNorm(d_model)
        self.ln2_begin = nn.LayerNorm(d_model)
        self.ff_begin  = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(p_drop),
        )
        self.TRAINABLE_PARAMS.append("multi_head_attention_layer")
        self.TRAINABLE_PARAMS.append("ff_begin")
        self.TRAINABLE_PARAMS.append("ln1_begin")
        self.TRAINABLE_PARAMS.append("ln2_begin")
        
        #self.TRAINABLE_PARAMS.append("visual.proj.weight")
        
            

    @property
    def num_devices(self) -> int:
        if not hasattr(self, "__device"):
            self.__device = torch.cuda.device_count()
        return self.__device

    def apply_lora(self):
        ## Use Lora ##
        use_lora = True
        num_of_layers_lora = 5
        use_up_layers = False
        print("Number of Vision Layers = ", self.vision_layer_num )
        total_layers = self.vision_layer_num
        if use_lora:
            if use_up_layers:
                start_layer = 0
                end_layer = num_of_layers_lora          
            else:
                start_layer = total_layers - num_of_layers_lora
                end_layer = total_layers
            for layer_index, layer in enumerate(self.visual.transformer.resblocks):
                if layer_index >= start_layer and layer_index < end_layer:
                    print(f"Applying LoRA to layer {layer_index}")
                    # Assuming `apply_lora` is a function to inject LoRA into a layer
                    layer.lora_addition()
                    layer.use_lora = True


    def pad_if_necessary(self, x: Tensor) -> Tensor:
        if not self.parallel_text_encoder:
            return x, 0

        n = x.size(0)
        if n % self.num_devices == 0:
            return x, 0

        pad = self.num_devices - n % self.num_devices
        return torch.cat([x, torch.zeros(pad, *x.shape[1:], device=x.device)], dim=0), pad

    def unpad_if_necessary(self, x: Tensor, pad: int) -> Tensor:
        if pad == 0:
            return x

        return x[:-pad]
    
    def v2v_use(self):
        for layer_index, layer in enumerate(self.visual.transformer.resblocks):
            if layer_index >= 0:
                layer.convert_to_v2v()

    def _default_encode_text(self, class_names: List[str]) -> Tensor:
        prompts = [self.template.format(name) for name in class_names]
        tokenized_text = clip.tokenize(prompts).cuda(non_blocking=True)
        text_features = super().encode_text(tokenized_text, batch_first=True)
        return text_features.unsqueeze(1)

    def _encode_text_2(self, prefix: Tensor, prompt: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        #### Burası Deneme ####
        #prompt_new, _ = self.multi_head_attention_layer(prompt,prompt,prompt)
        #prompt = prompt + prompt_new
        
        
        #prompt_new, _ = self.multi_head_attention_layer(self.ln1_begin(prompt),self.ln1_begin(prompt),self.ln1_begin(prompt))
        #prompt = prompt + prompt_new
        #prompt = prompt + self.ff_begin(self.ln2_begin(prompt))
        ########################
        x = torch.cat([prefix, prompt, suffix], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND  # This is not needed as we are using batch_first=True
        x, padding = self.pad_if_necessary(x)
        x, *_ = self.transformer(x, batch_first=True)
        x = self.unpad_if_necessary(x, padding)
        # x = x.permute(1, 0, 2)  # LND -> NLD  # This is not needed as we are using batch_first=True
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_tokens + self.n_token_context] @ self.text_projection
        return x
    
    def _encode_text(self, prefix: Tensor, prompt: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        #### Burası Deneme ####
        #prompt, _ = self.multi_head_attention_layer(prompt,prompt,prompt)
        ########################
        x = torch.cat([prefix, prompt, suffix], dim=1)
        x = x + self.positional_embedding.type(self.dtype)
        # x = x.permute(1, 0, 2)  # NLD -> LND  # This is not needed as we are using batch_first=True
        x, padding = self.pad_if_necessary(x)
        x, *_ = self.transformer(x, batch_first=True)
        x = self.unpad_if_necessary(x, padding)
        # x = x.permute(1, 0, 2)  # LND -> NLD  # This is not needed as we are using batch_first=True
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eot_tokens + self.n_token_context] @ self.text_projection
        return x

    def _single_forward_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        n_prompts = prompts.size(1)
        n_classes = prefix.size(0)


        text_features = self._encode_text_2(
            prefix.repeat_interleave(n_prompts, dim=0),
            prompts.reshape(prompts.shape[0] * prompts.shape[1] , prompts.shape[2], prompts.shape[3]),
            suffix.repeat_interleave(n_prompts, dim=0),
            eot_tokens.repeat_interleave(n_prompts),
        )
        text_features = text_features.unflatten(0, (n_classes, n_prompts))
        #print("Text Feature shape", text_features.shape)
        return text_features

    def _single_forward_encode_text2(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        n_prompts = prompts.size(0)
        n_classes = prefix.size(0)


        text_features = self._encode_text(
            prefix.repeat_interleave(n_prompts, dim=0),
            prompts.repeat(n_classes, 1, 1),
            suffix.repeat_interleave(n_prompts, dim=0),
            eot_tokens.repeat_interleave(n_prompts),
        )
        text_features = text_features.unflatten(0, (n_classes, n_prompts))
        #print("Text Feature shape", text_features.shape)
        return text_features
    
    def _loop_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        text_features = []
        for i in range(prompts.size(0)):
            x = self._encode_text(prefix, prompts[i : i + 1].expand(prefix.size(0), -1, -1), suffix, eot_tokens)
            text_features.append(x)

        return torch.stack(text_features, dim=1)

    def _most_efficient_encode_text(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        if self.parallel_text_encoder:
            return self._single_forward_encode_text(prefix, prompts, suffix, eot_tokens)
        return self._loop_encode_text(prefix, prompts, suffix, eot_tokens)
    
    def _most_efficient_encode_text2(self, prefix: Tensor, prompts: Tensor, suffix: Tensor, eot_tokens: Tensor) -> Tensor:
        if self.parallel_text_encoder:
            return self._single_forward_encode_text2(prefix, prompts, suffix, eot_tokens)
        return self._loop_encode_text(prefix, prompts, suffix, eot_tokens)

    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        if not self.learn_global_prompt and not self.learn_local_prompts:
            text_features = self._default_encode_text(class_names)
            return text_features, text_features

        tokenized_text = clip.tokenize(class_names).cuda(non_blocking=True)

        
        eot_tokens = tokenized_text.argmax(dim=-1)


        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)


        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]





        if self.learn_global_prompt or self.n_global_prompts > 1:
            global_prompt = self.global_prompt

          
            
            if self.prompts_batch_size < self.n_global_prompts and self.training:
                idx_select = torch.randperm(self.n_global_prompts)[: self.prompts_batch_size]  # we don't want to do this for local prompts
                global_prompt = self.global_prompt[idx_select]

              

            text_features = self._most_efficient_encode_text2(prefix, global_prompt, suffix, eot_tokens)
        else:
            text_features = self._default_encode_text(class_names)

        if self.learn_local_prompts or self.n_local_prompts > 1:
            ### IF USE Hierarchycal Prompts ###
            if self.use_hierarchycal_prompts:
                #local_prompts = torch.concat((self.global_prompt,self.local_prompts),dim=1)
                #local_text_features = self._most_efficient_encode_text(prefix, local_prompts, suffix[:,:-self.n_token_context,:], eot_tokens)
                local_prompts = self.global_prompt.mean(0) + self.local_prompts
                
                
                local_text_features = self._most_efficient_encode_text(prefix, local_prompts, suffix, eot_tokens)
                
            else:
                #global_cat = self.global_prompt[0].unsqueeze(0).unsqueeze(0).repeat(self.local_prompts.shape[0],self.local_prompts.shape[1],1,1)
                #local_prompts = torch.concat((global_cat,self.local_prompts),dim=2)
                #local_text_features = self._most_efficient_encode_text(prefix, local_prompts, suffix, eot_tokens)
                local_text_features = self._most_efficient_encode_text(prefix, self.local_prompts, suffix, eot_tokens)
                #local_text_features = self._most_efficient_encode_text2(prefix, self.local_prompts, suffix, eot_tokens)
        else:
            local_text_features = text_features
        

        return text_features, local_text_features

    def encode_image_and_proj(self, image: Tensor) -> Tuple[Tensor, Tensor]:
        image_features, local_features = self.encode_image(image)


        local_features = self.local_proj(local_features)


        ## Yeni Eklendi Dikkat ###

        #img_features_new = torch.concat((image_features.unsqueeze(1),local_features),dim=1)
        
        #image_features = image_features + self.last_attn(img_features_new, img_features_new, img_features_new, need_weights=False, attn_mask=None)[0][:,0,:]
        ###########################

        if hasattr(self.visual, "proj"):
            image_features = image_features @ self.visual.proj
            if self.use_local_features:
                local_features = local_features @ self.visual.proj

        
        return image_features, local_features

    def forward(
        self,
        image: Tensor,
        class_names: Optional[List[str]] = None,
        text_features: Optional[Tensor] = None,
        local_text_features: Optional[Tensor] = None,
    ) -> Tensor:
        if class_names is not None:
            assert isinstance(class_names, list), "class_names must be a list of strings"
        if text_features is not None:
            assert isinstance(text_features, torch.Tensor), "text_features must be a Tensor"
        assert class_names is not None or text_features is not None, "Please provide either class_names or text_features"

        if text_features is None:
            assert local_text_features is None, "local_text_features should be None if text_features is None"
            text_features, local_text_features = self.encode_text(class_names)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True) if self.learn_local_prompts else text_features

        #print("Image Shape", image.shape)
        image = image #+ torch.bmm(self.B,self.A)
        image_features, local_features = self.encode_image_and_proj(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        #print("Text Features last shape",text_features.shape)
        #print("Global image gfeat shape", image_features.shape)

        if self.learn_visual_prompt:
            global_logits = torch.einsum("bmd,kmd-> bkm", image_features, text_features)
        else:
            global_logits = torch.einsum("bd,kmd-> bkm", image_features, text_features)
        #print("Global logits shape", global_logits.shape)

        if self.use_local_features:
            local_features = local_features / local_features.norm(dim=-1, keepdim=True)
            local_logits = torch.einsum("bpd,knd-> bpkn", local_features, local_text_features)
            #local_logits = torch.einsum("bpd,knd-> bpkn", local_features, text_features)
            
            #print("Local feats shape", local_features.shape)
            #print("Local text shape", local_text_features.shape)
            #print("Local logits shape",local_logits.shape)
        else:
            local_logits = None

        return global_logits, local_logits

    def _prompt_features(self, promtps: Tensor) -> Tensor:
        tokenized_text = clip.tokenize("").cuda(non_blocking=True)
        eot_tokens = tokenized_text.argmax(dim=-1)

        with torch.no_grad():
            token_embeddings = self.token_embedding(tokenized_text)

        prefix = token_embeddings[:, :1, :]
        suffix = token_embeddings[:, 1 : -self.n_token_context, :]

        text_features = self._most_efficient_encode_text(prefix, promtps, suffix, eot_tokens)
        return text_features

    def prompt_features(
        self,
    ) -> Tensor:
        global_prompt_features = local_prompt_features = None
        if self.learn_global_prompt:
            global_prompt_features = self._prompt_features(self.global_prompt)

        if self.learn_local_prompts:
            local_prompt_features = self._prompt_features(self.local_prompts)

        return global_prompt_features, local_prompt_features

    @property
    def device(self) -> torch.device:
        return self.text_projection.device

    def freeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = False

        for module in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            module.eval()
            module.train = lambda _: None

        ### SONRADAN EKLENDI ###
        #self.visual.proj.requires_grad = True # Açarsan overfitliyor

    def unfreeze_clip(self) -> NoneType:
        for name, p in self.named_parameters():
            if not any([name.startswith(param) for param in self.TRAINABLE_PARAMS]):
                p.requires_grad = True

        for _ in filter(lambda m: isinstance(m, nn.BatchNorm2d), self.modules()):
            print("Warning this module has Batchnorm that cannot be unfrozen.")
            break

    def trainable_state_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.state_dict().items() if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])}

    def load_trainable_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> _IncompatibleKeys:
        keys = self.load_state_dict(state_dict, strict=False)
        missing_keys = [k for k in keys.missing_keys if any([k.startswith(param) for param in self.TRAINABLE_PARAMS])]
        if strict:
            error_msgs: List[str] = []
            if len(keys.unexpected_keys) > 0:
                error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in keys.unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys)))

            if len(error_msgs) > 0:
                raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(self.__class__.__name__, "\n\t".join(error_msgs)))

        return _IncompatibleKeys(missing_keys=missing_keys, unexpected_keys=keys.unexpected_keys)

    def orthonormal_rows(self,n_rows: int, d: int, *, device, dtype, generator=None):
        """
        Returns a (n_rows, d) matrix whose rows are orthonormal.
        Requires n_rows <= d (can't have >d orthonormal vectors in d-dim).
        Uses QR on a (d, n_rows) random matrix for numerical stability.
        """
        if n_rows > d:
            raise ValueError(
                f"Cannot create {n_rows} orthonormal vectors in {d}-D space. "
                "Reduce n_local_prompts or increase transformer_width."
            )
        # Random (d x n_rows) then QR -> Q (d x n_rows), columns orthonormal
        A = torch.randn(d, n_rows, device=device, dtype=dtype, generator=generator)
        # Reduced QR is fine; Q has shape (d, n_rows)
        Q, R = torch.linalg.qr(A, mode="reduced")
        # Make sign deterministic by forcing positive diagonal on R
        diag = torch.diag(R)
        signs = torch.sign(diag)
        signs[signs == 0] = 1
        Q = Q * signs  # broadcast on columns
        return Q.T  # (n_rows, d), rows orthonormal


    @torch.no_grad()
    def gram_schmidt_init_local_prompts_(self,local_prompts: nn.Parameter, T: int, *,
                                         scale: float = 1.0,
                                         generator: torch.Generator | None = None):
        """
        Orthonormal init across the n_local_prompts axis for the first T tokens.

        local_prompts: Parameter of shape [num_classes, n_local_prompts, n_token_context, d]
        T            : number of leading tokens to initialize (e.g., template length)
        scale        : scales the orthonormal rows (e.g., 0.02 to mimic CLIP std)
        """
        if not isinstance(local_prompts, (nn.Parameter, torch.Tensor)):
            raise TypeError("local_prompts must be a torch.nn.Parameter or Tensor")

        device = local_prompts.device
        dtype  = local_prompts.dtype

        C, P, Nc, d = local_prompts.shape
        if T > Nc:
            raise ValueError(f"T={T} exceeds n_token_context={Nc}")

        if P > d:
            raise ValueError(
                f"n_local_prompts={P} > transformer_width={d}. "
                "Cannot make prompts orthonormal in that space."
            )

        for c in range(C):
            for t in range(T):
                # Fill slice [P, d] with orthonormal rows
                ortho = self.orthonormal_rows(P, d, device=device, dtype=dtype, generator=generator)
                local_prompts[c, :, t, :].copy_(scale * ortho)

        return local_prompts  # for chaining if you like
    @torch.no_grad()
    def initialize_prompt(self) -> NoneType:
        if not self.learn_global_prompt and not self.learn_local_prompts:
            return

        template = self.template.replace("{}", " ").replace("_", " ").strip()
        tokenized_template = clip.tokenize(template)
        embedding = self.token_embedding(tokenized_template).type(self.dtype)
        global_prompt_init = embedding[:, 1 : 1 + self.template_init_tokens, :]

        if self.learn_global_prompt:
            #self.global_prompt.data[:, :, : self.template_init_tokens].copy_(global_prompt_init.clone().expand(self.num_classes, self.n_global_prompts, -1, -1))
            self.global_prompt.data[:, : self.template_init_tokens].copy_(global_prompt_init.clone().expand(self.n_global_prompts, -1, -1))

        if self.learn_local_prompts:
            #self.local_prompts.data[:, : self.template_init_tokens].copy_(global_prompt_init.clone().expand(self.n_local_prompts, -1, -1))
            self.local_prompts.data[:, :, : self.template_init_tokens].copy_(global_prompt_init.clone().expand(self.num_classes, self.n_local_prompts, -1, -1))
            T = self.template_init_tokens
            if self.init_method == "random":
                nn.init.xavier_uniform_(self.local_prompts[:,:, :T, :])
                #nn.init.xavier_uniform_(self.local_prompts[:, :T, :])
            else:
                g = torch.Generator(device=self.local_prompts.device).manual_seed(1)
                self.gram_schmidt_init_local_prompts_(self.local_prompts, T, scale=0.001, generator=g)
            #nn.init.xavier_uniform_(self.local_prompts[:, :T, :])
    def compute_gl_scores_deneme1(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> NoneType:
        
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        #global_probs = torch.softmax(global_logits  * logit_scale, dim=-1).cpu().numpy()
        scores = -np.max(global_probs, axis=-1)

        if local_logits is not None:
            B = local_logits.shape[0]
            local_logits = local_logits.topk(dim=1, k=10)[0]
            #global_weights = global_weighter(global_logits_orig,local_logits)
            #local_logits = (local_logits + global_weights) / 2
            ######################
            local_logits = local_logits.permute(1,3,0,2).contiguous() 
            M = local_logits.shape[0]
            N = local_logits.shape[1]
            n_cls = local_logits.shape[3]
            b = local_logits.shape[2]
            ### Local Logits shape token_size, number of text tokens 4, batch_size, number of class
            local_logits = local_logits.view(local_logits.shape[0],local_logits.shape[1], local_logits.shape[2] * local_logits.shape[3])
            local_logits = local_logits.permute(2,0,1)
            
            wdist = 1.0 - local_logits
            xx=torch.zeros(b*n_cls, M, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / M)
            yy=torch.zeros(b*n_cls, N, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / N)
            with torch.no_grad():
                KK = torch.exp(-wdist / 0.1)
                T = self.Sinkhorn(KK,xx,yy)
                #T = self.sinkhorn_unbalanced(KK,xx,yy)
           

            #local_logits = T * local_logits
            #local_logits = local_logits.reshape(B,1000,10,4)
            local_loss = torch.sum(T * local_logits, dim=(1,2))
            #local_loss = local_loss.mean(1)
            local_loss = local_loss.contiguous().view(b,n_cls)
            
            #local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            local_probs = torch.softmax(local_loss  / self.ood_temp_scale , dim=-1).cpu().numpy()
            #local_probs = torch.softmax(local_loss * logit_scale , dim=-1).cpu().numpy()
            #print(local_probs.shape)
            #local_score = -np.max(local_probs, axis=(1, 2))
            local_score = -np.max(local_probs, axis=(1))
        
            scores += local_score

        return scores

    
    def compute_gl_scores_deneme2(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> NoneType:
        
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        #global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        #global_probs = torch.softmax(global_logits  * logit_scale, dim=-1).cpu().numpy()
        #scores = -np.max(global_probs, axis=-1)

        if local_logits is not None:
            B = local_logits.shape[0]
            #local_logits = local_logits.topk(dim=1, k=10)[0]
            mean_for_ranking = local_logits.mean(dim=-1)        # (B, N, C)

            # 2) Top-k along N using the mean
            maxk = int(max(self.topk))                          # or just your desired k
            mean_topk_vals, topk_idx = torch.topk(
                mean_for_ranking, k=maxk, dim=1, largest=True, sorted=True
            )                                                    # shapes: (B, K, C), (B, K, C)

            # 3) Use the SAME indices to gather from the original 4-channel tensor
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, CH)  # (B, K, C, 4)
            local_logits = torch.gather(local_logits, dim=1, index=idx_expanded)  # (B, K, C, 4)

            #global_weights = global_weighter(global_logits_orig,local_logits)
            #local_logits = (local_logits + global_weights) / 2
            ######################
            local_logits = local_logits.permute(1,3,0,2).contiguous() 
            M = local_logits.shape[0]
            N = local_logits.shape[1]
            n_cls = local_logits.shape[3]
            b = local_logits.shape[2]
            ### Local Logits shape token_size, number of text tokens 4, batch_size, number of class
            local_logits = local_logits.view(local_logits.shape[0],local_logits.shape[1], local_logits.shape[2] * local_logits.shape[3])
            local_logits = local_logits.permute(2,0,1)
            
            wdist = 1.0 - local_logits
            xx=torch.zeros(b*n_cls, M, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / M)
            yy=torch.zeros(b*n_cls, N, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / N)
            with torch.no_grad():
                KK = torch.exp(-wdist / 0.1)
                T = self.Sinkhorn(KK,xx,yy)
                #T = self.sinkhorn_unbalanced(KK,xx,yy)
           

            #local_logits = T * local_logits
            #local_logits = local_logits.reshape(B,1000,10,4)
            #local_loss = torch.sum(T * local_logits, dim=(1))
            local_loss = torch.sum(local_logits, dim=(1))
            local_loss = local_loss.mean(1)
            local_loss = local_loss.contiguous().view(b,n_cls)
            
            #local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            #local_probs = torch.softmax(local_loss  / self.ood_temp_scale , dim=-1).cpu().numpy()
            #local_probs = torch.softmax(local_loss * logit_scale , dim=-1).cpu().numpy()
            #print(local_probs.shape)
            #local_score = -np.max(local_probs, axis=(1, 2))
            #local_score = -np.max(local_probs, axis=(1))
            total_probs = torch.softmax((local_loss + global_logits) / 2  / self.ood_temp_scale , dim=-1).cpu().numpy()
        
            #scores += local_score
        scores = -np.max(total_probs, axis=(1))

        return scores

    def compute_gl_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        scores = -np.max(global_probs, axis=-1)

        if local_logits is not None:
            local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
            local_score = -np.max(local_probs, axis=(1, 2))
            scores += local_score

        return scores

    def compute_L_mcm_scores(
        self,
        local_logits: Tensor,
    ) -> NoneType:
        assert local_logits is not None
        local_probs = torch.softmax(local_logits.mean(dim=-1) / self.ood_temp_scale, dim=-1).cpu().numpy()
        local_score = -np.max(local_probs, axis=(1, 2))
        return local_score

    def compute_mcm_scores(
        self,
        global_logits: Tensor,
    ) -> NoneType:
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(global_logits / self.ood_temp_scale, dim=-1).cpu().numpy()
        global_score = -np.max(global_probs, axis=-1)
        return global_score

    def compute_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
        ood_method: Optional[str] = None,
    ) -> NoneType:
        if ood_method is None:
            ood_method = self.ood_method

        if ood_method == "GL-MCM":
            return self.compute_gl_scores(global_logits, local_logits)
        elif ood_method == "MCM":
            return self.compute_mcm_scores(global_logits)
        elif ood_method == "L-MCM":
            return self.compute_L_mcm_scores(local_logits)
        else:
            raise ValueError(f"Method {self.ood_method} not implemented")

    @torch.no_grad()
    def create_prediction_scores(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            local_logits = vlp_tools.topk_reduce(local_logits, topk=self.topk)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs

    @torch.no_grad()
    def create_prediction_scores2(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            topk = [5,5,5,5]
            local_logits = vlp_tools.topk_reduce(local_logits, topk=topk)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs

    @torch.no_grad()
    def create_prediction_scores3(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            #topk = [5,5,5,5]
            sofmaxed_local_logits = torch.softmax(logit_scale * local_logits, dim=2)
            local_logits = vlp_tools.topk_reduce_deneme(local_logits,sofmaxed_local_logits, topk=self.topk)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs

    def create_prediction_scores4(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits_orig = global_logits
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            global_min, _ = global_logits_orig.min(dim=1, keepdim=True)
            global_max, _ = global_logits_orig.max(dim=1, keepdim=True)
            global_weights = (global_logits_orig - global_min) / (global_max - global_min + 1e-8)
            global_weights = global_weights.mean(dim=-1)
            local_logits = local_logits * global_weights.unsqueeze(1).unsqueeze(-1).repeat(1,local_logits.shape[1],1,local_logits.shape[-1])
            local_logits = vlp_tools.topk_reduce(local_logits, topk=self.topk)
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs
    
    def create_prediction_scores5(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits_orig = global_logits
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)
        total_local_probs = []

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            global_weights = global_weighter(global_logits_orig,local_logits)
            local_part_1 = (local_logits[:,:,:,-1] + global_weights[:,:,:,-1]) / 2
            local_part_2 = (local_logits[:,:,:,-2] + local_part_1) / 2
            local_part_3 = (local_logits[:,:,:,-3] + local_part_2) / 2
            local_part_4 = (local_logits[:,:,:,-4] + local_part_3) / 2
            #local_part_1 = (local_logits[:,:,:,-1] + global_weights[:,:,:,-1]) / 2
            #local_part_2 = (local_logits[:,:,:,-2] + global_weights[:,:,:,-1]) / 2
            #local_part_3 = (local_logits[:,:,:,-3] + global_weights[:,:,:,-1]) / 2
            #local_part_4 = (local_logits[:,:,:,-4] + global_weights[:,:,:,-1]) / 2
            local_logits = torch.concat([local_part_4.unsqueeze(-1),local_part_3.unsqueeze(-1),local_part_2.unsqueeze(-1),local_part_1.unsqueeze(-1)],dim=-1)
            
            local_logits = vlp_tools.topk_reduce(local_logits, topk=self.topk)
            
            total_local_probs.append(torch.softmax(logit_scale * local_logits[:,:,-1], dim=-1))
            total_local_probs.append(torch.softmax(logit_scale * local_logits[:,:,-2], dim=-1))
            total_local_probs.append(torch.softmax(logit_scale * local_logits[:,:,-3], dim=-1))
            total_local_probs.append(torch.softmax(logit_scale * local_logits[:,:,-4], dim=-1))
            local_logits = local_logits.mean(dim=-1)
            local_probs = torch.softmax(logit_scale * local_logits, dim=-1)
            
            gl_logits = (global_logits + local_logits) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs, total_local_probs
    
    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def sinkhorn_unbalanced(self, K, u, v):
        """
        The Unbalanced Sinkhorn algorithm.

        Args:
            K (torch.Tensor): The Gibbs kernel, K = exp(-C/eps).
            u (torch.Tensor): The marginals of the first distribution.
            v (torch.Tensor): The marginals of the second distribution.

        Returns:
            torch.Tensor: The computed transport plan T.
        """
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        self.reg_m = 100
        self.eps = 0.1
        # Pre-compute the exponent for the update steps.
        # This is the core modification for unbalanced transport.
        power = self.reg_m / (self.reg_m + self.eps)
        
        thresh = 1e-2
        for _ in range(100):
            r0 = r
            
            # --- Modified Sinkhorn Updates ---
            # Update r based on c
            r = (u / (K @ c.unsqueeze(-1)).squeeze(-1))**power
            
            # Update c based on the new r
            # K.transpose(-2, -1) is equivalent to K.permute(0, 2, 1) for 3D tensors
            c = (v / (K.transpose(-2, -1) @ r.unsqueeze(-1)).squeeze(-1))**power
            
            # Check for convergence
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        # Compute the final transport plan T
        T = r.unsqueeze(-1) * K * c.unsqueeze(-2)
        return T
    
    def create_prediction_scores_last(
        self,
        global_logits: Tensor,
        local_logits: Optional[Tensor],
    ) -> Tensor:
        logit_scale = self.logit_scale.exp()
        global_logits_orig = global_logits
        global_logits = global_logits.mean(dim=-1)
        global_probs = torch.softmax(logit_scale * global_logits, dim=-1)
        total_local_probs = []

        if local_logits is None:
            local_probs = None
            gl_probs = global_probs
        else:
            
            #local_logits = local_logits.sum(dim=1).mean(dim=-1)
            ### Yeni Ekledim ###
            maxk = max(self.topk)
            B, N, C, CH = local_logits.shape
            mean_for_ranking = local_logits.mean(dim=-1)        # (B, N, C)

            # 2) Top-k along N using the mean
            maxk = int(max(self.topk))                          # or just your desired k
            mean_topk_vals, topk_idx = torch.topk(
                mean_for_ranking, k=maxk, dim=1, largest=True, sorted=True
            )                                                    # shapes: (B, K, C), (B, K, C)

            # 3) Use the SAME indices to gather from the original 4-channel tensor
            idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, -1, CH)  # (B, K, C, 4)
            local_logits = torch.gather(local_logits, dim=1, index=idx_expanded)  # (B, K, C, 4)
            #local_logits = local_logits.topk(dim=1, k=maxk)[0]
            #global_weights = global_weighter(global_logits_orig,local_logits)
            #local_logits = (local_logits + global_weights) / 2
            ######################
            local_logits = local_logits.permute(1,3,0,2).contiguous() 
            M = local_logits.shape[0]
            N = local_logits.shape[1]
            n_cls = local_logits.shape[3]
            b = local_logits.shape[2]
            ### Local Logits shape token_size, number of text tokens 4, batch_size, number of class
            local_logits = local_logits.view(local_logits.shape[0],local_logits.shape[1], local_logits.shape[2] * local_logits.shape[3])
            local_logits = local_logits.permute(2,0,1)
            
            wdist = 1.0 - local_logits
            xx=torch.zeros(b*n_cls, M, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / M)
            yy=torch.zeros(b*n_cls, N, dtype=local_logits.dtype, device=local_logits.device).fill_(1. / N)
            with torch.no_grad():
                KK = torch.exp(-wdist / 0.1)
                T = self.Sinkhorn(KK,xx,yy)
                #T = self.sinkhorn_unbalanced(KK,xx,yy)
           

            local_loss = torch.sum(T * local_logits, dim=(1)) 
            
            local_loss = local_loss.mean(1)
            #local_loss = local_loss / 6.0
            local_loss = local_loss.contiguous().view(b,n_cls)
            
            
            
            local_probs = torch.softmax(logit_scale  * local_loss, dim=-1)
            
            gl_logits = (global_logits + local_loss) / 2
            gl_probs = torch.softmax(logit_scale * gl_logits, dim=-1)

        return gl_probs, global_probs, local_probs
