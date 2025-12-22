import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .tools.wan_model import WanModel


class HFT5Encoder:
    """Wrapper for HuggingFace T5 encoder, compatible with original T5EncoderModel interface"""
    def __init__(self, text_len, dtype=torch.float32, device=torch.device("cpu"), model_name="google/umt5-base"):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        
        print(f"Loading {model_name} from HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, 
            dtype=dtype
        ).encoder  # Only use the encoder part
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device)
    
    def __call__(self, texts, device):
        """Encode texts, returns list of tensors (one per text, with padding removed)"""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_len,
            return_tensors="pt"
        )
        ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        
        # Encode (model should already be on device via external .model.to(device) call)
        context = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        
        # Get sequence lengths (excluding padding)
        seq_lens = mask.sum(dim=1).long()
        
        # Return list of tensors with padding removed (same as original T5EncoderModel)
        return [u[:v] for u, v in zip(context, seq_lens)]


class DiffForcingWanModel(nn.Module):
    def __init__(
        self,
        model_name="google/umt5-base",  # HuggingFace model name
        input_dim=256,
        hidden_dim=1024,
        ffn_dim=2048,
        freq_dim=256,
        num_heads=8,
        num_layers=8,
        time_embedding_scale=1.0,
        chunk_size=5,
        noise_steps=10,
        use_text_cond=True,
        text_len=512,
        drop_out=0.1,
        cfg_scale=5.0,
        prediction_type="vel",  # "vel", "x0", "noise"
        causal=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.time_embedding_scale = time_embedding_scale
        self.chunk_size = chunk_size
        self.noise_steps = noise_steps
        self.use_text_cond = use_text_cond
        self.drop_out = drop_out
        self.cfg_scale = cfg_scale
        self.prediction_type = prediction_type
        self.causal = causal

        self.text_dim = 768  # umt5-base hidden size
        self.text_len = text_len
        self.model_name = model_name
        
        # Load model and tokenizer from HuggingFace
        print(f"Loading {model_name} from HuggingFace...")
        self.text_encoder = HFT5Encoder(
            text_len=text_len,
            dtype=torch.float32,
            device=torch.device("cpu"),
            model_name=model_name,
        )

        # Text encoding cache
        self.text_cache = {}
        self.model = WanModel(
            model_type="t2v",
            patch_size=(1, 1, 1),
            text_len=self.text_len,
            in_dim=self.input_dim,
            dim=self.hidden_dim,
            ffn_dim=self.ffn_dim,
            freq_dim=self.freq_dim,
            text_dim=self.text_dim,
            out_dim=self.input_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
            causal=self.causal,
        )
        self.param_dtype = torch.float32

    def encode_text_with_cache(self, text_list, device):
        """Encode text using cache
        Args:
            text_list: List[str], list of texts
            device: torch.device
        Returns:
            List[Tensor]: List of encoded text features
        """
        text_features = []
        indices_to_encode = []
        texts_to_encode = []

        # Check cache
        for i, text in enumerate(text_list):
            if text in self.text_cache:
                # Get from cache and move to correct device
                cached_feature = self.text_cache[text].to(device)
                text_features.append(cached_feature)
            else:
                # Need to encode
                text_features.append(None)
                indices_to_encode.append(i)
                texts_to_encode.append(text)

        # Batch encode uncached texts
        if texts_to_encode:
            self.text_encoder.model.to(device)
            encoded = self.text_encoder(texts_to_encode, device)

            # Store in cache and update results
            for idx, text, feature in zip(indices_to_encode, texts_to_encode, encoded):
                # Cache to CPU to save GPU memory
                self.text_cache[text] = feature.cpu()
                text_features[idx] = feature

        return text_features

    def preprocess(self, x):
        # (bs, T, C) -> (bs, C, T, 1, 1)
        x = x.permute(0, 2, 1)[:, :, :, None, None]
        return x

    def postprocess(self, x):
        # (bs, C, T, 1, 1) ->  (bs, T, C)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(x.size(0), x.size(2), -1)
        return x

    def _get_noise_levels(self, device, seq_len, time_steps):
        """Get noise levels"""
        # noise_level[i] = clip(1 + i / chunk_size - time_steps, 0, 1)
        noise_level = torch.clamp(
            1
            + torch.arange(seq_len, device=device) / self.chunk_size
            - time_steps.unsqueeze(1),
            min=0.0,
            max=1.0,
        )
        return noise_level

    def add_noise(self, x, noise_level):
        """Add noise
        Args:
            x: (B, T, D)
            noise_level: (B, T)
        """
        noise = torch.randn_like(x)
        # noise_level: (B, T) -> (B, T, 1)
        noise_level = noise_level.unsqueeze(-1)
        noisy_x = x * (1 - noise_level) + noise_level * noise
        return noisy_x, noise

    def forward(self, x):
        feature = x["feature"]  # (B, T, C)
        feature_length = x["feature_length"]  # (B,)
        batch_size, seq_len, _ = feature.shape
        device = feature.device

        # Randomly use a time step
        time_steps = []
        for i in range(batch_size):
            valid_len = feature_length[i].item()
            # Random float from 0 to valid_len/chunk_size, not an integer
            max_time = valid_len / self.chunk_size
            # max_time = valid_len / self.chunk_size + 1
            time_steps.append(torch.FloatTensor(1).uniform_(0, max_time).item())
        time_steps = torch.tensor(time_steps, device=device)  # (B,)
        noise_level = self._get_noise_levels(device, seq_len, time_steps)  # (B, T)

        # # Debug: Print noise levels
        # print("Time steps and corresponding noise levels:")
        # for i in range(batch_size):
        #     t = time_steps[i].item()
        #     # Get noise level at each position
        #     start_idx = int(self.chunk_size * (t - 1))
        #     end_idx = int(self.chunk_size * t) + 2
        #     # Limit to valid range
        #     start_idx = max(0, start_idx)
        #     end_idx = min(seq_len, end_idx)
        #     print(time_steps[i])
        #     print(noise_level[i, start_idx:end_idx])

        # Add noise to entire sequence
        noisy_feature, noise = self.add_noise(feature, noise_level)  # (B, T, D)

        # Debug: Print noise addition information
        # print("Added noise levels at chunk positions:")
        # for i in range(batch_size):
        #     t = time_steps[i].item()
        #     start_idx = int(self.chunk_size * (t - 1))
        #     end_idx = int(self.chunk_size * t) + 2
        #     # Limit to valid range
        #     start_idx = max(0, start_idx)
        #     end_idx = min(seq_len, end_idx)
        #     test1 = (
        #         feature[i, start_idx:end_idx, :] - noisy_feature[i, start_idx:end_idx, :]
        #     )
        #     test2 = (
        #         noise[i, start_idx:end_idx, :] - noisy_feature[i, start_idx:end_idx, :]
        #     )
        #     # Compute length on last dimension
        #     print(test1.norm(dim=-1))
        #     print(test2.norm(dim=-1))

        feature = self.preprocess(feature)  # (B, C, T, 1, 1)
        noisy_feature = self.preprocess(noisy_feature)  # (B, C, T, 1, 1)
        noise = self.preprocess(noise)  # (B, C, T, 1, 1)

        feature_ref = []
        noise_ref = []
        noisy_feature_input = []
        for i in range(batch_size):
            t = time_steps[i].item()
            end_index = int(self.chunk_size * t) + 1
            valid_len = feature_length[i].item()
            end_index = min(valid_len, end_index)
            feature_ref.append(feature[i, :, :end_index, ...])
            noise_ref.append(noise[i, :, :end_index, ...])
            noisy_feature_input.append(noisy_feature[i, :, :end_index, ...])

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                text_end_list = x["feature_text_end"]
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    if np.random.rand() > self.drop_out:
                        single_text_end_list = [0] + [
                            min(t, seq_len) for t in single_text_end_list
                        ]
                    else:
                        single_text_list = [""]
                        single_text_end_list = [0, seq_len]
                    single_text_length_list = [
                        t - b
                        for t, b in zip(
                            single_text_end_list[1:], single_text_end_list[:-1]
                        )
                    ]
                    single_text_context = self.encode_text_with_cache(
                        single_text_list, device
                    )
                    single_text_context = [
                        u.to(self.param_dtype) for u in single_text_context
                    ]
                    for u, duration in zip(
                        single_text_context, single_text_length_list
                    ):
                        all_text_context.extend([u for _ in range(duration)])
                    all_text_context.extend(
                        [
                            single_text_context[-1]
                            for _ in range(seq_len - single_text_end_list[-1])
                        ]
                    )
            else:
                all_text_context = [
                    (u if np.random.rand() > self.drop_out else "") for u in text_list
                ]
                all_text_context = self.encode_text_with_cache(all_text_context, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        # Through WanModel
        predicted_result = self.model(
            noisy_feature_input,
            noise_level * self.time_embedding_scale,
            all_text_context,
            seq_len,
            y=None,
        )  # (B, C, T, 1, 1)

        loss = 0.0
        for b in range(batch_size):
            if self.prediction_type == "vel":
                vel = feature_ref[b] - noise_ref[b]  # (C, input_length, 1, 1)
                squared_error = (
                    predicted_result[b][:, -self.chunk_size :, ...]
                    - vel[:, -self.chunk_size :, ...]
                ) ** 2
            elif self.prediction_type == "x0":
                squared_error = (
                    predicted_result[b][:, -self.chunk_size :, ...]
                    - feature_ref[b][:, -self.chunk_size :, ...]
                ) ** 2
            elif self.prediction_type == "noise":
                squared_error = (
                    predicted_result[b][:, -self.chunk_size :, ...]
                    - noise_ref[b][:, -self.chunk_size :, ...]
                ) ** 2
            sample_loss = squared_error.sum().mean()
            loss += sample_loss
        loss = loss / batch_size

        loss_dict = {"total": loss, "mse": loss}
        return loss_dict

    def generate(self, x, num_denoise_steps=None):
        """
        Generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        # # debug
        # x["text"] = [["walk forward.", "sit down.", "stand up."] for _ in range(batch_size)]
        # x["feature_text_end"] = [[1, 2, 3] for _ in range(batch_size)]
        # text = x["text"]
        # text_end = x["feature_text_end"]
        # print(text)
        # print(text_end)
        # print(batch_size, seq_len, self.chunk_size)

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)  # (B, C, T, 1, 1)

        # Calculate total number of time steps needed
        max_t = 1 + (seq_len - 1) / self.chunk_size

        # Step size for each advancement
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                generated_length = []
                text_end_list = x["feature_text_end"]
                full_text = []
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    single_text_end_list = [0] + [
                        min(t, seq_len) for t in single_text_end_list
                    ]
                    generated_length.append(single_text_end_list[-1])
                    single_text_length_list = [
                        t - b
                        for t, b in zip(
                            single_text_end_list[1:], single_text_end_list[:-1]
                        )
                    ]
                    full_text.append(
                        " ////////// ".join(
                            [
                                f"{u} //dur:{t}"
                                for u, t in zip(
                                    single_text_list, single_text_length_list
                                )
                            ]
                        )
                    )
                    single_text_context = self.encode_text_with_cache(
                        single_text_list, device
                    )
                    single_text_context = [
                        u.to(self.param_dtype) for u in single_text_context
                    ]
                    for u, duration in zip(
                        single_text_context, single_text_length_list
                    ):
                        all_text_context.extend([u for _ in range(duration)])
                    all_text_context.extend(
                        [
                            single_text_context[-1]
                            for _ in range(
                                seq_len + self.chunk_size - single_text_end_list[-1]
                            )
                        ]
                    )
            else:
                generated_length = feature_length
                full_text = text_list
                all_text_context = self.encode_text_with_cache(text_list, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            generated_length = feature_length
            full_text = [""] * batch_size
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        # Get empty text condition encoding (for CFG)
        text_null_list = [""] * batch_size
        text_null_context = self.encode_text_with_cache(text_null_list, device)
        text_null_context = [u.to(self.param_dtype) for u in text_null_context]

        # print(len(all_text_context), len(text_null_context))

        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # Current time step
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            # Calculate current noise schedule
            noise_level = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_text_context,
                seq_len + self.chunk_size,
                y=None,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    text_null_context,
                    seq_len + self.chunk_size,
                    y=None,
                )  # (B, C, T, 1, 1)
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

        generated = self.postprocess(generated)  # (B, T, C)
        y_hat_out = []
        for i in range(batch_size):
            # cut off the padding
            single_generated = generated[i, : generated_length[i], :]
            y_hat_out.append(single_generated)
        out = {}
        out["generated"] = y_hat_out
        out["text"] = full_text

        return out

    @torch.no_grad()
    def stream_generate(self, x, num_denoise_steps=None):
        """
        Streaming generation - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """
        feature_length = x["feature_length"]
        batch_size = len(feature_length)
        seq_len = max(feature_length).item()

        # # debug
        # x["text"] = [["walk forward.", "sit down.", "stand up."] for _ in range(batch_size)]
        # x["feature_text_end"] = [[1, 2, 3] for _ in range(batch_size)]
        # text = x["text"]
        # text_end = x["feature_text_end"]
        # print(text)
        # print(text_end)
        # print(batch_size, seq_len, self.chunk_size)

        if num_denoise_steps is None:
            num_denoise_steps = self.noise_steps
        assert num_denoise_steps % self.chunk_size == 0

        device = next(self.parameters()).device

        # Initialize entire sequence as pure noise
        generated = torch.randn(
            batch_size, seq_len + self.chunk_size, self.input_dim, device=device
        )
        generated = self.preprocess(generated)  # (B, C, T, 1, 1)

        # Calculate total number of time steps needed
        max_t = 1 + (seq_len - 1) / self.chunk_size

        # Step size for each advancement
        dt = 1 / num_denoise_steps
        total_steps = int(max_t / dt)

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str] or List[List[str]]
            if isinstance(text_list[0], list):
                generated_length = []
                text_end_list = x["feature_text_end"]
                full_text = []
                all_text_context = []
                for single_text_list, single_text_end_list in zip(
                    text_list, text_end_list
                ):
                    single_text_end_list = [0] + [
                        min(t, seq_len) for t in single_text_end_list
                    ]
                    generated_length.append(single_text_end_list[-1])
                    single_text_length_list = [
                        t - b
                        for t, b in zip(
                            single_text_end_list[1:], single_text_end_list[:-1]
                        )
                    ]
                    full_text.append(
                        " ////////// ".join(
                            [
                                f"{u} //dur:{t}"
                                for u, t in zip(
                                    single_text_list, single_text_length_list
                                )
                            ]
                        )
                    )
                    single_text_context = self.encode_text_with_cache(
                        single_text_list, device
                    )
                    single_text_context = [
                        u.to(self.param_dtype) for u in single_text_context
                    ]
                    for u, duration in zip(
                        single_text_context, single_text_length_list
                    ):
                        all_text_context.extend([u for _ in range(duration)])
                    all_text_context.extend(
                        [
                            single_text_context[-1]
                            for _ in range(
                                seq_len + self.chunk_size - single_text_end_list[-1]
                            )
                        ]
                    )
            else:
                generated_length = feature_length
                full_text = text_list
                all_text_context = self.encode_text_with_cache(text_list, device)
                all_text_context = [u.to(self.param_dtype) for u in all_text_context]
        else:
            generated_length = feature_length
            full_text = [""] * batch_size
            all_text_context = [""] * batch_size
            all_text_context = self.encode_text_with_cache(all_text_context, device)
            all_text_context = [u.to(self.param_dtype) for u in all_text_context]

        # Get empty text condition encoding (for CFG)
        text_null_list = [""] * batch_size
        text_null_context = self.encode_text_with_cache(text_null_list, device)
        text_null_context = [u.to(self.param_dtype) for u in text_null_context]

        # print(len(all_text_context), len(text_null_context))

        commit_index = 0
        # Progressively advance from t=0 to t=max_t
        for step in range(total_steps):
            # Current time step
            t = step * dt
            start_index = max(0, int(self.chunk_size * (t - 1)) + 1)
            end_index = int(self.chunk_size * t) + 1
            time_steps = torch.full((batch_size,), t, device=device)

            # Calculate current noise schedule
            noise_level = self._get_noise_levels(
                device, seq_len + self.chunk_size, time_steps
            )  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(batch_size):
                noisy_input.append(generated[i, :, :end_index, ...])

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                all_text_context,
                seq_len + self.chunk_size,
                y=None,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    text_null_context,
                    seq_len + self.chunk_size,
                    y=None,
                )  # (B, C, T, 1, 1)
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated[i, :, start_index:end_index, ...] += predicted_vel * dt

            if commit_index < start_index:
                output = generated[:, :, commit_index:start_index, ...]
                output = self.postprocess(output)  # (B, T, C)
                y_hat_out = []
                for i in range(batch_size):
                    if commit_index < generated_length[i]:
                        y_hat_out.append(
                            output[i, : generated_length[i] - commit_index, ...]
                        )
                    else:
                        y_hat_out.append(None)

                out = {}
                out["generated"] = y_hat_out
                yield out
                commit_index = start_index

        output = generated[:, :, commit_index:, ...]
        output = self.postprocess(output)  # (B, T_remain, C)
        y_hat_out = []
        for i in range(batch_size):
            if commit_index < generated_length[i]:
                y_hat_out.append(output[i, : generated_length[i] - commit_index, ...])
            else:
                y_hat_out.append(None)
        out = {}
        out["generated"] = y_hat_out
        yield out

    def init_generated(self, seq_len, batch_size=1, num_denoise_steps=None):
        self.seq_len = seq_len
        self.batch_size = batch_size
        if num_denoise_steps is None:
            self.num_denoise_steps = self.noise_steps
        else:
            self.num_denoise_steps = num_denoise_steps
        assert self.num_denoise_steps % self.chunk_size == 0
        self.dt = 1 / self.num_denoise_steps
        self.current_step = 0
        self.text_condition_list = [[] for _ in range(self.batch_size)]
        self.generated = torch.randn(
            self.batch_size, self.seq_len * 2 + self.chunk_size, self.input_dim
        )
        self.generated = self.preprocess(self.generated)  # (B, C, T, 1, 1)
        self.commit_index = 0

    @torch.no_grad()
    def stream_generate_step(self, x, first_chunk=True):
        """
        Streaming generation step - Diffusion Forcing inference
        Uses triangular noise schedule, progressively generating from left to right

        Generation process:
        1. Start from t=0, gradually increase t
        2. Each t corresponds to a noise schedule: clean on left, noisy on right, gradient in middle
        3. After each denoising step, t increases slightly and continues
        """

        device = next(self.parameters()).device
        if first_chunk:
            self.generated = self.generated.to(device)

        # Encode text condition (using cache)
        if self.use_text_cond and "text" in x:
            text_list = x["text"]  # List[str]
            new_text_context = self.encode_text_with_cache(text_list, device)
            new_text_context = [u.to(self.param_dtype) for u in new_text_context]
        else:
            new_text_context = [""] * self.batch_size
            new_text_context = self.encode_text_with_cache(new_text_context, device)
            new_text_context = [u.to(self.param_dtype) for u in new_text_context]

        # Get empty text condition encoding (for CFG)
        text_null_list = [""] * self.batch_size
        text_null_context = self.encode_text_with_cache(text_null_list, device)
        text_null_context = [u.to(self.param_dtype) for u in text_null_context]

        for i in range(self.batch_size):
            if first_chunk:
                self.text_condition_list[i].extend(
                    [new_text_context[i]] * self.chunk_size
                )
            else:
                self.text_condition_list[i].extend([new_text_context[i]])

        end_step = (
            (self.commit_index + self.chunk_size)
            * self.num_denoise_steps
            / self.chunk_size
        )
        while self.current_step < end_step:
            current_time = self.current_step * self.dt
            start_index = max(0, int(self.chunk_size * (current_time - 1)) + 1)
            end_index = int(self.chunk_size * current_time) + 1
            time_steps = torch.full((self.batch_size,), current_time, device=device)

            noise_level = self._get_noise_levels(device, end_index, time_steps)[
                :, -self.seq_len :
            ]  # (B, T)

            # Predict noise through WanModel
            noisy_input = []
            for i in range(self.batch_size):
                noisy_input.append(
                    self.generated[i, :, :end_index, ...][:, -self.seq_len :]
                )  # (C, T, 1, 1)

            text_condition = []
            for i in range(self.batch_size):
                text_condition.extend(
                    self.text_condition_list[i][:end_index][-self.seq_len :]
                )  # (T, D, 4096)

            # print("////////////////////")
            # print("current step: ", self.current_step)
            # print("chunk size: ", self.chunk_size)
            # print("start_index: ", start_index)
            # print("end_index: ", end_index)
            # print("noisy_input shape: ", noisy_input[0].shape)
            # print("noise_level: ", noise_level[0, start_index:end_index])
            # print("text_condition shape: ", len(text_condition))
            # print("commit_index: ", self.commit_index)
            # print("////////////////////")

            predicted_result = self.model(
                noisy_input,
                noise_level * self.time_embedding_scale,
                text_condition,
                min(end_index, self.seq_len),
                y=None,
            )  # (B, C, T, 1, 1)

            # Adjust using CFG
            if self.cfg_scale != 1.0:
                predicted_result_null = self.model(
                    noisy_input,
                    noise_level * self.time_embedding_scale,
                    text_null_context,
                    min(end_index, self.seq_len),
                    y=None,
                )  # (B, C, T, 1, 1)
                predicted_result = [
                    self.cfg_scale * pv - (self.cfg_scale - 1) * pvn
                    for pv, pvn in zip(predicted_result, predicted_result_null)
                ]

            for i in range(self.batch_size):
                predicted_result_i = predicted_result[i]  # (C, input_length, 1, 1)
                if end_index > self.seq_len:
                    predicted_result_i = torch.cat(
                        [
                            torch.zeros(
                                predicted_result_i.shape[0],
                                end_index - self.seq_len,
                                predicted_result_i.shape[2],
                                predicted_result_i.shape[3],
                                device=device,
                            ),
                            predicted_result_i,
                        ],
                        dim=1,
                    )
                if self.prediction_type == "vel":
                    predicted_vel = predicted_result_i[:, start_index:end_index, ...]
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "x0":
                    predicted_vel = (
                        predicted_result_i[:, start_index:end_index, ...]
                        - self.generated[i, :, start_index:end_index, ...]
                    ) / (
                        noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
                elif self.prediction_type == "noise":
                    predicted_vel = (
                        self.generated[i, :, start_index:end_index, ...]
                        - predicted_result_i[:, start_index:end_index, ...]
                    ) / (
                        1
                        + self.dt
                        - noise_level[i, start_index:end_index]
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    self.generated[i, :, start_index:end_index, ...] += (
                        predicted_vel * self.dt
                    )
            self.current_step += 1
        output = self.generated[:, :, self.commit_index : self.commit_index + 1, ...]
        output = self.postprocess(output)  # (B, 1, C)
        out = {}
        out["generated"] = output
        self.commit_index += 1

        if self.commit_index == self.seq_len * 2:
            self.generated = torch.cat(
                [
                    self.generated[:, :, self.seq_len :, ...],
                    torch.randn(
                        self.batch_size,
                        self.input_dim,
                        self.seq_len,
                        1,
                        1,
                        device=device,
                    ),
                ],
                dim=2,
            )
            self.current_step -= self.seq_len * self.num_denoise_steps / self.chunk_size
            self.commit_index -= self.seq_len
            for i in range(self.batch_size):
                self.text_condition_list[i] = self.text_condition_list[i][
                    self.seq_len :
                ]
        return out
