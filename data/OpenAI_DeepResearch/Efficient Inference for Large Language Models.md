# Efficient Inference for Large Language Models

## 1 Introduction

Large Language Models (LLMs) are advanced AI systems trained on massive text corpora to generate or understand human language. Modern LLMs often consist of hundreds of millions to trillions of parameters, making them extremely powerful but also computationally heavy to run. Inference refers to using a trained model to make predictions or generate text from new inputs. Efficient inference has become a critical concern as LLMs grow larger - the latency, memory footprint, and cost of deploying these models can be prohibitive without optimizations. For example, GPT- 3 (175 billion parameters, introduced in 2020) demonstrated remarkable capabilities but highlighted the challenges of serving such a model: each query requires passing the input through many layers and billions of weight multiplications 1 . Even with powerful hardware, a naive GPT- 3 inference can take several seconds per output token and consume enormous GPU memory.

Improving inference efficiency means delivering results faster and with less computational resource usage, without significantly degrading the model' s accuracy or utility. This survey provides a comprehensive overview of techniques and developments that enable efficient inference for LLMs. We cover formal definitions of key concepts, historical milestones in the field, foundational methods (like model compression and architecture improvements), the latest algorithms for speeding up inference, system- level optimizations, and emerging challenges. The goal is to make these topics accessible to newcomers while offering depth and references for advanced researchers. We highlight how efficient inference methods are empowering new applications - from deploying personal Al assistants on mobile devices to serving global- scale chatbots - and what open problems remain in balancing model size and speed.

Scope and Organization: We begin by defining LLMs and discussing why inference is traditionally slow for large models. Next, we present a brief historical timeline of major milestones in efficient LLM inference. The core of the survey is organized into sections on: (1) Model Compression techniques (quantization, distillation, pruning) that shrink or simplify models; (2) Efficient Model Architectures (such as sparse or adaptive models that reduce computation per token); (3) Inference Algorithm Improvements (like caching and speculative decoding to reduce latency); and (4) Systems and Hardware Optimizations (software frameworks, parallelism, and hardware accelerators for LLM serving). We then discuss applications across subfields (e.g. NLP, vision, edge computing) that benefit from these techniques. Finally, we outline open challenges and future directions in the quest to make LLMs faster and more efficient.

## 2 Background: Key Concepts and Definitions

Large Language Models (LLMs): In NLP, a language model is a system that learns the probability distribution of sequences of words and can predict text. An LLM refers to a language model with very high capacity (typically hundreds of millions or more parameters) trained on vast amounts of text 2 . These models are usually based on the Transformer architecture 3 - particularly the decoder- only Transformers for generation (such as GPT series). LLMs acquire statistical knowledge of language through self- supervised training (e.g. next- word prediction) and can be adapted or prompted to perform various tasks like answering questions, summarizing, coding, etc. Formally, an LLM defines a probability

distribution  $\{P(w\_ 1, \dots, w\_ n)\}$  over sequences of tokens and during inference produces the most likely (or sample) next token given a context.

Inference vs Training: Training a model involves updating its parameters on data and is typically done once (or occasionally to fine- tune) at high computational cost. Inference is the act of using the trained model to produce outputs for new inputs. Inference for LLMs is computationally intensive because it often involves evaluating dozens or hundreds of neural network layers and performing billions of multiply- add operations for a single input. Unlike training, inference is often expected to happen in real- time (e.g. responding to a user query), so latency is crucial. Efficiency in this context means optimizing the forward- pass computations of the model. Key metrics include: latency (time to produce a result for a single input), throughput (how many outputs can be produced per second or per hardware unit), and memory usage (especially the RAM/VRAM required to hold model weights and activations).

Why LLM Inference is Challenging: Modern LLMs owe much of their power to their size - e.g., GPT- 3 with 175B parameters or recent open models with  $70\mathsf{B}+$  parameters. This size directly translates to heavy computation and memory requirements. Running inference on a 175B model in 16- bit floating point precision requires loading 350 billion bytes  $(\approx 350~\mathrm{GB})$  of weight data, which far exceeds the memory of a single GPU. Even when the model fits, each forward pass involves multiplying huge matrices (e.g. a single Transformer feed- forward layer might be thousands of neurons wide) which takes significant time. Moreover, Transformer- based LLMs have self- attention mechanisms with quadratic complexity in the sequence length - generating each token requires attention over all previous tokens, making long prompts or outputs slow to handle. The process of text generation is inherently sequential: the model generates one token at a time, and cannot easily parallelize the production of the next token since each depends on the previously generated tokens 1 . This leads to underutilization of hardware (GPUs may be idle waiting for sequential steps) and high inter- token latency. All these factors mean that a naive deployment of an LLM can be extremely slow and costly, motivating a wide range of research into efficiency.

Efficiency Metrics: When we talk about "efficient inference," it usually implies reducing one or more of the following:

- Latency per token 
- e.g. using faster algorithms so each token is generated quicker (important for interactive use). 
- Throughput 
- e.g. serving more requests in parallel or in batch on the same hardware. 
- Memory footprint 
- e.g. compressing the model so it can run on smaller devices or more models can be loaded on a server. 
- Compute (FLOPs) 
- reducing the number of floating point operations needed, which often correlates with both speed and energy usage. 
- Energy consumption 
- using fewer joules of energy per inference (often tied to FLOPs and memory access patterns).

Different applications prioritize different metrics: a real- time chatbot on a phone values low latency and memory, whereas a cloud inference service might prioritize high throughput to serve many users.

Formal Definitions: We define a few terms for clarity:

- Model Size: The number of parameters (weights) in the model. An order of magnitude guide: "large" often means billions of parameters. 
- Precision: The numerical precision used for weights/activations (e.g. FP32, FP16, int8). Lower precision can speed up inference if hardware supports it, at the risk of approximation error. 
- Quantization: The process of reducing precision (e.g. from 32-bit floats to 8-bit integers) for model weights or activations to reduce memory and compute. 
- Pruning: Removing some model weights (e.g. setting them to zero or entirely eliminating connections/

nodes) to make the model sparser and smaller.

- Knowledge Distillation: Training a smaller "student" model to replicate the behavior of a large "teacher" model, thereby transferring knowledge and achieving a smaller model with relatively high performance 4. 
- Throughput-oriented inference: Batch-processing many inputs simultaneously to maximize total tokens per second (often used in server settings). Latency-oriented inference: optimizing for single-stream speed (often for interactive or real-time use). 
- Speedup: When we say a method yields, for example,  $2 \times$  faster inference," it means latency (or throughput) is improved such that the model generates outputs in half the time (or double the outputs in same time) compared to baseline.

With these concepts established, we next examine the historical evolution of efficient inference techniques for LLMs, before diving into the methods in detail.

## 3 Historical Milestones in Efficient LLM Inference

The pursuit of making inference faster and lighter has paralleled the rapid growth in model sizes. Below is a timeline of key milestones and developments:

- 2015 
- Model Compression Pioneered: Early works on compressing neural networks (for computer vision and small models) introduced techniques like pruning and quantization. Han et al.'s Deep Compression (2015) showed that significant fractions of weights could be pruned and the rest quantized with minimal accuracy loss, foreshadowing later compression of LLMs. Hinton et al. (2015) introduced knowledge distillation as a way to compress models by teaching a smaller model to mimic a larger one's outputs.

- 2017 
- Transformer Architecture: The introduction of the Transformer model (Vaswani et al., "Attention is All You Need") revolutionized NLP. Transformers enabled training much larger models by scaling efficiently on parallel hardware, but also brought the challenge of quadratic-time self-attention. This marks the point where models started becoming "large" in today's sense. Efficient attention mechanisms (to handle long sequences faster) became an active research area subsequently.

- 2018 
- BERT and Accelerated Inference: Google's BERT (Bidirectional Encoder Representations from Transformers) became a landmark LLM with 340M parameters (BERT-large). Soon after, knowledge distillation was applied to compress BERT 
- yielding DistilBERT in 2019, which retained  $\sim 97\%$  of BERT' a accuracy on language understanding while being  $40\%$  smaller and  $60\%$  faster in inference 4. This validated distillation as an effective compression tool for LLMs.

- 2019 
- GPT-2 and Emergence of Gigantic Models: OpenAI's GPT-2 (1.5B params) garnered attention both for its text generation ability and for initial concerns about deploying such a powerful model. The model was eventually released and ran on consumer GPUs, but it underscored that even larger models were on the horizon.

- 2020 
- GPT-3 (175B) released: GPT-3 dramatically scaled up language models to 175 billion parameters, demonstrating new capabilities ("few-shot" learning from prompts) but at enormous computational cost. This sparked intense interest in inference optimization, as deploying GPT-3 required multi-GPU systems and incurred high latency. Research into 8-bit precision for Transformers followed: e.g., the GPT-3.int8() work showed that 8-bit weight quantization can be applied even to 175B models with negligible accuracy loss 5. Hardware

vendors also began adding support for mixed- precision and int8 matrix operations (NVIDIA's Ampere GPUs, 2020, introduced INT8 Tensor Cores).

- 2021 
- Mixture-of-Experts (MoE) and Sparse Models: Google's Switch Transformer (2021) and related MoE models demonstrated that a trillion-parameter model could be trained by activating only a small subset of weights per token. For example, Switch used 2048 experts but routed each input to just one expert, achieving the quality of a dense trillion-parameter model at a fraction of the compute per inference 6 7. This renewed interest in sparsity for efficiency - using sparse MoEs or pruning weights to skip unnecessary computations.

- 2022 
- Efficient Inference Libraries: Microsoft's DeepSpeed-Inference (2022) introduced ZeRO-Inference technology to leverage CPU and NVMe memory for hosting large models and streaming weights to GPUs on the fly 8. This allowed, for instance, running a 530B parameter model on a single GPU by offloading most of the model to CPU, albeit with some latency cost. Also in 2022, the concept of FlashAttention was introduced 9 - an optimized attention algorithm that reduces memory access and speeds up attention computation (important for long sequences). On the model compression front, Frantar et al. (2022) proposed SparseGPT, showing that GPT-like models can be pruned post hoc (after training) to  $\sim 50\%$  sparsity in one shot with minimal loss, enabling potential speedups on hardware that can exploit sparsity.

- 2023 
- 4-bit Quantization & LLMs Everywhere: Meta's LLaMA models (released early 2023) ranged up to 65B parameters and, being open-source, unleashed a wave of experimentation in the community on compressing and deploying LLMs. Techniques for 4-bit quantization such as GPTQ emerged, allowing weights to be quantized to just 4 bits per parameter with surprisingly little performance drop 10 11. This made it feasible to run multi-billion-parameter models on a single GPU or even on CPU. Indeed, 2023 saw LLMs like LLaMA-7B running on ordinary laptops and smartphones through 4-bit or 8-bit compression. However, it was also observed that quantization doesn't always speed up inference unless the hardware and software are optimized for low precision; oftentimes memory savings are the main benefit 12.

- 2023 
- Speculative Decoding and Serving Optimizations: In mid-2023, new decoding algorithms were introduced to overcome the sequential bottleneck. Speculative decoding is one such method (reported by OpenAI and others) where a smaller "draft" model generates multiple tokens in one go which the large model then verifies in one batch 13 14. This can accelerate generation by  $2 - 3\times$  without quality loss by effectively parallelizing token production. On the systems side, researchers from UC Berkeley released vLLM (2023), an open-source high-throughput inference engine that uses continuous batching and PagedAttention memory management 15 16. By packing user requests together dynamically and storing attention keyvalues in a paged, memory-efficient manner, vLLM achieved up to  $24\times$  higher throughput than naive approaches 17.

- 2024 and Beyond 
- Towards Universal Efficiency: By 2024, efficient LLM inference is a fast-moving target. New hardware accelerators specifically for Transformers (with support for 4-bit and sparse computation) are emerging. Open-source models like Mistral 7B (2023) demonstrated that smaller models can sometimes reach parity with larger ones via clever training, reducing the need for brute-force size. Research continues on long-context LLMs (handling inputs of tens of thousands of tokens efficiently) and adaptive computation (models that adjust their depth or computation based on input difficulty). The community is also coalescing around standard benchmarks for inference efficiency (e.g., MLPerf Inference added LLM tasks, including an MoE benchmark 18). We anticipate further advances like 2-bit quantization (with training

adjustments), better support for unstructured sparsity in hardware, and algorithms for partial generation parallelism – all aiming to make LLM deployment more accessible and sustainable.

Having set the stage, we will now delve into the major categories of techniques that enable these milestones – from compressing the models themselves to optimizing the runtime execution.

## 4 Model Compression Techniques for Inference Efficiency

One broad approach to efficient inference is model compression – making the model smaller or simpler so that each inference requires less computation. Importantly, compression is typically done offline (before deployment) and aims to preserve as much of the model's accuracy as possible. We discuss three major compression strategies: quantization, knowledge distillation, and pruning. Each of these reduces the inference cost in a different way.

### 4.1 Quantization of Model Weights and Activations

Quantization reduces the number of bits used to represent each number (weight or activation) in the model. By using lower precision arithmetic, we shrink the memory footprint and can leverage faster lower- precision instructions on hardware. A common form is going from 32- bit floating point (FP32) to 8- bit integers (int8). This alone gives a  $4 \times$  reduction in memory and potentially a  $4 \times$  speedup in math operations, if the hardware supports int8 math at equal throughput. Modern accelerators (GPUs, TPUs) indeed have specialized units for 8- bit matrix multiplies, so int8 inference has become a standard for LLMs  $^{19}$ .

Is quantization possible without hurting accuracy? Early attempts in the 2010s on smaller models found that naive quantization (especially below 8 bits) could significantly degrade model accuracy due to rounding errors. However, recent research has developed advanced quantization schemes that maintain accuracy even for very large models. One key insight is to handle outlier weights or activations carefully. For example, GPT- 3's layers have a few extremely large magnitude weights that, if quantized poorly, can wreck performance  $^{11}$ . Techniques like LLM.int8() (Dettmers et al. 2022) introduced an 8- bit matrix multiply that reserves extra bits for such outliers (by effectively keeping a small fraction of weights in higher precision)  $^{5}$ . This enabled 8- bit inference on models up to 175B with negligible accuracy loss  $^{21}$ .

Researchers have since pushed to 4- bit and even 2- bit weights. A 4- bit model is  $8 \times$  smaller than its FP32 original, a tremendous savings. A 2023 comprehensive study by Jin et al. found that 4- bit quantized LLMs can retain comparable performance to full precision on many benchmarks  $^{10}$ . They did note, however, that quantization can sometimes slow down decoding on current hardware  $^{12}$ , because existing GPU kernels are not fully optimized for 4- bit arithmetic and because of overheads in handling those 4- bit packed formats. Nonetheless, methods like GPTQ (GPT Quantization) have gained popularity for post- training quantization: GPTQ does a layer- by- layer quantization, optimizing for the least error by using a small calibration set and even leveraging second- order information (like approximating Hessian) to decide the best rounding for each weight  $^{22}$ . The result is that a model like LLaMA- 65B can be compressed to 4- bit and still answer questions almost as well as the original. Going below 4 bits is more challenging – 2- bit quantization often incurs noticeable degradation unless the model is fine- tuned with quantization- aware training.

Another aspect is quantizing activations (the intermediate outputs for each layer). Weight quantization alone reduces model size, but activation maps during inference can also be large. Quantizing activations to 8- bit is common in computer vision models, but for LLMs, activation distributions vary with each token

input. Techniques distinguish between static quantization (calibrating activation ranges ahead of time, possibly per layer) and dynamic quantization (determining the scale for activations on the fly as data passes through) 24. Many LLM deployments use a mix: weights in 4- bit or 8- bit, and activations in 8- bit or 16- bit to be safe. Research into minimizing the precision without loss is ongoing. As of 2025, 8- bit weight/activation inference is routine, 4- bit weights with 8- bit activations is achievable for many models, and anything lower usually requires special fine- tuning or hybrid schemes (e.g. some parts in higher precision).

In summary, quantization offers a straightforward trade- off: memory and compute are reduced at the cost of introducing quantization error. The success of methods like LLM.int8 and GPTQ shows that for LLMs, much of the redundancy can be squeezed out in lower bits. Hardware support is key - future GPUs with 4- bit matrix engines or FPGAs/ASICs optimized for low precision will unlock the full speed potential of quantization. Table 1 (conceptual) illustrates common quantization levels and their typical impact:

- 16-bit (e.g. FP16/BF16): Baseline for many models; no significant loss, half the memory of FP32.  
- 8-bit (int8):  $4 \times$  smaller; minimal accuracy loss with proper calibration; well-supported on GPUs (TensorRT, ONNX Runtime int8).  
- 4-bit (int4):  $8 \times$  smaller; requires advanced techniques (GPTQ, etc.) to maintain accuracy; speedup depends on kernel support.  
- 2-bit / binary:  $16 \times$  or  $32 \times$  smaller; active research, usually large accuracy drop except perhaps for specific layers or if retrained.

### 4.2 Knowledge Distillation (Teacher-Student Models)

While quantization reduces precision, knowledge distillation reduces the model's size (number of parameters) by training a smaller model to imitate a larger one. The idea, introduced by Hinton et al. (2015), is that a teacher model (the original large LLM, or an ensemble of models) generates "soft" targets - e.g., probability distributions over next tokens or classifications - and a student model is trained to match those outputs. The student is typically much smaller (e.g. a quarter or half the number of layers or hidden units). By learning to approximate the teacher's function, the student can achieve close performance with far less computation.

This technique was famously applied to compress BERT: DistilBERT (Sahn et al. 2019) took a 110M parameter BERT- base model down to 66M (40% fewer parameters) and achieved 60% faster inference while retaining ~97% of BERT's accuracy on GLUE benchmarks 4. DistilBERT's success showed that for many tasks, the extra layers in BERT were not all crucial - a smaller Transformer could be almost as good if trained with the right guidance. Similarly, TinyBERT and MiniLM were other distilled versions that preserved task performance with smaller sizes.

For generative LLMs like GPT, distillation has also been explored. E.g., researchers distilled GPT- 3 (175B) down to ~20B or less to create more efficient variants for specific tasks. One challenge unique to LLMs is that they perform a variety of tasks (zero- shot or few- shot) that aren't easy to capture with one training objective. Nevertheless, task- specific distillation (for instance distilling a big model to a smaller one specialized in dialogue) is viable and used in industry to deploy models on moderate hardware. There are also techniques like sequence- level distillation, where the student learns to generate whole sequences (texts) that match the teacher's outputs, and iterative distillation (repeatedly distilling in stages to incrementally smaller models).

The trade- off in distillation: it requires an expensive additional training phase (running many inference passes of the teacher to generate training data for the student). However, once done, you get a model that is intrinsically faster - it has fewer layers or smaller layers, so each inference is cheaper. Unlike

quantization, which is just a different way of computing the same model, distillation actually creates a new model that might generalize slightly differently. Ideally, the only loss is a small drop in accuracy. In practice, careful tuning is needed: e.g. choosing the right size for the student, the right "temperature" for soft targets (to give richer information than 1- hot correct answers), and potentially combining the distillation loss with the original task loss.

Distillation can also be combined with quantization and other compression: e.g. you might distill a 100B model into a 10B model, then quantize that 10B to 8- bit, ending up with a model that is both smaller and uses low precision. One interesting emerging direction is multi- teacher or ensemble distillation - using multiple large models to teach one student, aiming to transfer a broader set of skills or multilingual capabilities into one compact model.

In summary, knowledge distillation is a powerful tool to compress the knowledge of a large model into a smaller one, directly reducing inference cost proportional to the size reduction. Its success in cases like DistilBERT  $4$  shows that significant speedups (e.g.  $1.5 - 2 \times$ ) are achievable with minimal impact on quality. For very large models, distillation remains challenging if one wants a general- purpose student (not just for a narrow task), but research continues into making "small" LLMs that maintain the versatility of their bigger teachers.

### 4.3 Pruning and Sparsity

Another avenue to make models efficient is pruning, which involves removing unnecessary weights or neurons from the model. Neural networks are often over- parameterized, and many weights have little effect on outputs. Pruning identifies these and sets them to zero (or removes them entirely), resulting in a sparse model. The advantage at inference is that if the hardware or software can skip the zero weights, it will do less work.

There are different granularity levels for pruning:

- Unstructured pruning: Remove individual weights. For example, eliminate  $50\%$  of the smallest-magnitude weights network-wide. This yields a model matrix with  $50\%$  zeros scattered in it. The challenge is that while mathematically the FLOPs are halved, not all hardware can exploit a random sparse pattern efficiently (some do, using specialized sparse libraries).

- Structured pruning: Remove entire neurons or attention heads or even layers. For instance, prune 2 out of 12 attention heads in each Transformer block, or remove blocks entirely. This yields a smaller dense model (which is easier for hardware to handle). It often requires fine-tuning to recover from the larger capacity drop when whole units are removed.

Historically, pruning deep networks could even improve generalization (by removing "noise" weights). For LLMs, pruning became feasible at scale only recently. In 2022, a method called SparseGPT demonstrated pruning GPT- like models in one shot (without retraining) to fairly high sparsity  $25$ . By making use of a clever weight update strategy (solving a local optimization for each layer' s weights to best retain output on a small sample), SparseGPT could remove  $50\%$  of weights from a 175B model with only minor increase in perplexity. Further iterative pruning could go beyond  $50\%$  with retraining. Another approach named Wanda (2023) prunes weights based on a combination of weight magnitude and an estimate of the activation (input) norm, achieving good one- shot sparsity as well  $26$ .

Pruning is appealing because a highly sparse model, if supported by the runtime, can reduce both memory and compute. For example, going to  $50\%$  zeros ideally gives  $2 \times$  speedup. Some hardware (like NVIDIA' s GPUs with 2:4 structured sparsity support) can natively double throughput if exactly  $50\%$  of

weights are zero in a specific pattern. Unstructured sparsity is less efficiently used, but libraries like TensorRT and ONE are starting to introduce support for it when sparsity is high enough.

There is also dynamic sparsity: instead of a fixed pruned model, make the model activate only a subset of weights per input. Mixture- of- Experts (discussed in the next section) is one form, but even standard models can be augmented with techniques like Dynamic Sparse Attention (only compute some attention scores) or conditional computation (e.g. early exiting from layers if output is confident). Early exit, for instance, allows the model to stop forward propagation for easy inputs, effectively skipping computations for those cases. This was explored in smaller models and could theoretically apply to LLMs to save time on tokens that are easy to predict.

In summary, pruning turns a dense LLM into a sparse one, potentially cutting down the operations needed. It can be done post- hoc (with slight accuracy loss) or during training (to fine- tune and regain accuracy). The main limitation is achieving actual speed gains: as hardware and software catch up to better utilize irregular sparse patterns, we expect pruning to play a greater role. Already, the possibility of pruning half of GPT- 3's weights for free has been shown 25; future models might be intentionally trained to be sparse from the start.

## 5 Efficient Model Architectures and Adaptations

Beyond compressing existing models, another approach is designing architectures that inherently require less computation at inference. This includes leveraging sparsity in the model structure, using alternative mechanisms to avoid expensive operations like dense attention, or splitting the model into parts that can be used more flexibly. We highlight a few key architecture- level strategies: Mixture- of- Experts (MoE) sparsity, efficient attention mechanisms for long contexts, and other adaptive structures.

### 5.1 Mixture-of-Experts (Sparse Transformers)

Mixture- of- Experts (MoE) models introduce sparsity at a high level: they consist of many sub- models ("experts"), but for any given input token only a small number of experts are activated. In a Transformer MoE layer, for example, instead of a single feed- forward network (FFN) there might be, say, 16 expert FFNs each with their own weights. A learned gating network routes each token to the top-  $\mathfrak{Sk}$  appropriate experts (often  $Sk = 1$  or 2) 7. The outputs are combined and passed on, and only those experts incurred computation. This means the model's parameter count can be enormous (sum of all experts' weights), but the compute per token is much lower than using all of them.

![](images/a854c6937bece1e6f393fe4119095b3f72bf96079a6460d226291166b53152c6.jpg)  
Figure 2: Illustration of a Switch Transformer encoder block. We replace the dense feed forward network (FFN) layer present in the Transformer with a sparse Switch FFN layer (light blue). The layer operates independently on the tokens in the sequence. We diagram two tokens ( $x_{1} =$  "More" and  $x_{2} =$  "Parameters" below) being routed (solid lines) across four FFN experts, where the router independently routes each token. The switch FFN layer returns the output of the selected FFN multiplied by the router gate value (dotted-line).

Illustration of a Switch Transformer MoE layer (based on Fedus et al., 2021) replacing a dense feed- forward network with multiple expert networks. A router assigns each token (e.g., "More" and "Parameters") to one of the expert FFNs, so each token only goes through a fraction of the total networks. 7 27

Google's Switch Transformer was a breakthrough example: it had up to 2048 experts and over a trillion parameters, yet during inference each token only used one expert (plus the router overhead), making its computational cost similar to a dense model of much smaller size 6 . The trade- off is that not all those parameters are utilized at once - they represent model capacity that can specialize on different tokens or contexts. MoE models thus are compute- efficient at inference if the gating works well, because you get the effect of a huge model' s quality while only computing a portion of it. In fact, the Switch Transformer paper reported faster per- token inference compared to a dense model of equal quality 6 .

Challenges with MoEs include: balancing the load (ensuring experts are evenly used), increased memory usage (you must store all experts in RAM/VRAM, even if not using them each time 28 ), and complexity in training. For inference specifically, serving MoEs at scale requires routing many tokens potentially to different devices if experts are sharded - this can introduce communication overhead. Recent work on inference- optimal MoEs tries to determine the best number of experts so that inference throughput is maximized without saturating routing bottlenecks 29 (e.g. too many experts might give diminishing returns). There are also techniques for expert parallelism (distributing experts across GPUs) and caching expert outputs for reuse when the same expert is repeatedly used.

Despite hurdles, MoEs are a promising way to have a "large model" that's mostly dormant until needed. Think of it as an ensemble where only one model from the ensemble is picked per input. Open- source MoEs like Mistral' s Mixtral- 8x7B (2023) have begun to appear, indicating practical viability. They report that MoE models can achieve the same quality as dense models much faster during training and also enjoy faster inference for the same parameter count 6 . A well- known downside is memory: having, say, 8 experts each of size 7B means 56B total parameters must be loaded in memory to serve the model, even if a given token only uses 7B of them 28 . Thus, MoEs trade off runtime compute for memory - a beneficial trade if you have ample RAM but limited compute per token.

In summary, MoEs introduce sparse computation: only part of the network "fires" for each input. This aligns with an intuitive idea that not every component of a huge model is needed for every task - instead,

specialists can be activated on demand. The result can be significantly lower inference cost for the same result quality, provided the routing overhead is managed. We might envision future LLM deployments where a giant pool of experts exists in a server's memory, and each query efficiently taps a small subset, enabling both quality and speed.

### 5.2 Efficient (Long) Attention Mechanisms

The self- attention in Transformers is powerful but costly: it scales  $O(\sin^{\wedge}2\sin)$  with sequence length  $\sin \xi$  . For long inputs (documents, multi- turn conversations, code files, etc.), the attention mechanism becomes a major bottleneck in inference. Thus, a line of research has produced efficient attention variants that approximate or restrict the attention computation to achieve better than quadratic scaling.

Some notable efficient attention approaches relevant to LLM inference:

- Sparse attention patterns: Models like Longformer (2020) and BigBird (2020) use attention that is not fully dense 
- each token attends only to a limited window of nearby tokens plus a few global tokens 30. This yields  $O(\sin \xi)$  or  $O(\sin \lambda \log n\xi)$  complexity, enabling inputs of thousands of tokens with manageable cost. At inference, such models don't blow up in computation as  $\sin \xi$  grows. The trade-off is that the model architecture has to be designed and trained with this pattern; you can't easily apply it post-hoc to a pretrained dense-attention model without fine-tuning. However, these have been successful in tasks like long document QA.

- Low-rank or kernel-based approximations: The idea here is that the full attention matrix (which is  $\sin \lambda$  times  $\sin \xi$ ) often has redundant information or can be approximated by lower-dimensional structures. Methods like Linformer (2020) project the length dimension to a smaller size before attention, making complexity linear in  $\sin \xi$ . Others like Performer (2021) use random feature methods to linearize the softmax attention (making it technically  $O(\sin \xi)$ ). These approaches allow scaling to longer sequences with less cost. At inference time, they can significantly speed up processing of long contexts, albeit sometimes at a minor loss in sequence modeling fidelity.

- Memory or recurrence-based models: Some recent models revisit recurrent architectures (which inherently handle long sequences by state passing) or use external memory to avoid full attention over long histories. Examples include the Reformer (which uses locality-sensitive hashing to sparsify attention) and various state-space models or hybrid Transformer-RNNs. While not mainstream for LLMs yet, they show potential to handle, say, tens of thousands of tokens efficiently, which could be crucial as context windows expand.

- Retrieval-based augmentation: A different take on efficiency is not architecture per se, but using retrieval of relevant snippets from a database such that the model doesn't need to internalize or attend over all knowledge. For instance, instead of a 100 billion parameter model storing all facts, a smaller model can query a search index for relevant text and then process it. This can be seen as efficiency (less internal parameters, shorter effective context that's relevant) at the cost of an external system. Retrieval-Augmented Generation (RAG) and related techniques reduce the need for extremely large LLMs by offloading part of the "knowledge" to a retriever. While this doesn't speed up the core model's matrix multiplications, it allows using a smaller model to achieve results comparable to a much larger one on knowledge-intensive tasks.

When it comes to long contexts, special mention goes to FlashAttention (Dao et al. 2022) - this isn't a new model, but a better implementation of standard attention. FlashAttention uses GPU- friendly tiling and on- the- fly computation to reduce memory usage from quadratic to linear, and avoid excessive memory reads/writes 9. Although it still does  $O(\sin^{\wedge}2\sin)$  computations, it is much faster in practice for moderate lengths because it maximizes throughput and keeps data in high- speed on- chip memory.

FlashAttention and its successors (like xFormers, and FlashAttention 2) are now often integrated in LLM inference frameworks to squeeze more speed out of attention, especially when using long contexts (e.g. 2k- 8k tokens).

In summary, efficient attention and related architectural tweaks address the sequence length scalability, which is a growing issue as LLMs move to handle longer documents or dialogues. By using sparsity, approximations, or better algorithms, these techniques enable faster inference on long inputs and/or allow models to consider more context without timing out. For instance, combining FlashAttention with a Longformer- style sparse windowed attention can allow a model to handle 4x longer input in roughly the same time as the original handled the shorter input – a big win for tasks like summarizing long articles. As LLM applications demand context lengths of 100k tokens (as some 2023 models advertise), such efficient attention mechanisms will be indispensable.

### 5.3 Adaptive Depth and Early Exiting

Another architectural idea is making the model's depth or computation adaptive to the input. Not every input token may need the full power of a 100- layer network to get a confident prediction. For classification tasks, early exit schemes have been proposed where intermediate layers have classifiers that can decide to output early if the prediction is already confident. For generation, this is trickier, but conceptually one could imagine that "easy" tokens (like completing a common phrase) might be decided with fewer decoding steps of internal layers.

While not yet common in large language generation, researchers have explored Dynamic Transformer architectures, where each token can have a different number of layers applied. For example, a token might carry a state that indicates it should pass through another block or not. If a token's representation is already good enough, it might skip subsequent layers (this can be gated by some learned threshold). This way, on simple inputs, the model does less work.

One concrete example in NLP is the Universal Transformer (2018) which allowed variable number of layer updates per token (though it was more about adaptive computation for different positions in a sequence). There are also Depth Adaptive Transformers (Elbayad et al. 2020) which learn to terminate early for sequence tasks. Applying these to LLMs could yield inference speedups when, say, generating very predictable continuations vs. when deep reasoning is needed. However, controlling quality is challenging – one doesn't want the model to prematurely exit and produce a wrong or simplistic output.

Speculative decoding (covered in the next section) can also be seen as a form of adaptive computation – using a cheap model for easy predictions and only falling back to the big model when needed.

In summary, adaptive architectures aim to avoid wasted computation on cases where the "full force" of the model isn't necessary. This remains a frontier area for LLMs, as most current large models still do a fixed amount of work per token. But with more research, conditional computation within the model could unlock huge efficiency gains by tailoring the compute to the input's complexity.

## 6 Inference-Time Algorithmic Optimizations

Even with a fixed model architecture and size, there are many ways to make the process of inference faster or more efficient. This section discusses techniques that operate at inference- time: using caching, smarter generation algorithms, batching strategies, etc., to reduce latency and computation. We cover

caching and reuse, speculative decoding, batching and parallelism, and how these contribute to efficient serving.

### 6.1 Caching and Reuse of Computations

One fundamental optimization when generating text with an LLM is to cache the intermediate results from previous tokens. In autoregressive generation (as with GPT models), each 'new token' s prediction involves recomputing the Transformer layers for the entire input sequence (which grows one token longer each time). However, the Transformer allows us to cache the key and value vectors from the self- attention for past tokens so that we don't recompute those from scratch at every step. This is known as the KV cache. Using the KV cache, each new token only requires computing attention of the new query vector against past keys, rather than recomputing keys for all past tokens. In practice, this transforms the time per token from growing with sequence length to being roughly constant (after an initial prompt). Without caching, generating a sequence of length  $\) 5\(would be O($ T^2\ $) in cost; with caching, it's O($ T$ (linear in the number of tokens, because each step is constant- time work).

All major LLM implementations use caching. However, storing and managing the cache efficiently is important. The cache is basically the model's "memory" of the conversation; for a 2048- token context and a model with, say, 50 layers and hidden size 1600, the KV cache can be several gigabytes. Efficient memory management like PagedAttention (from vLLM) breaks the cache into chunks and allows reusing memory across different request batches 31 32. This avoids fragmentation and lets servers handle many sequence generations without running out of memory. Some optimizations even allow sharing computation across sequences with overlapping prefixes (useful if many users prompt with similar starting text).

Another form of reuse is prompt precomputation. If we know part of the input in advance (for example, a system prompt that is always the same), we can run the model on that once and cache the final hidden state, then reuse it for every query. This saves doing those initial layers repeatedly. Libraries like Hugging Face Transformers support feeding a past key values argument exactly for this reason: you can concatenate a constant prefix's cache with the new query's cache.

There is also a possibility of reusing computation between different model queries via batching - if two users ask similar questions at the same time, a smart server might batch them together and do a single matrix multiply on a batch of 2 (which is more efficient on GPU than two separate multiplies). This ventures into the next topic of continuous batching.

### 6.2 Continuous Batching and Parallelism in Serving

Batching is the technique of processing multiple inputs simultaneously as one larger batch, utilizing the parallel nature of GPU hardware. For throughput- oriented scenarios, higher batch sizes lead to better utilization and thus more tokens per second overall. However, batching can increase latency for individual queries (since each might wait for others to be grouped and processed together).

Traditional batching is done in fixed batches - e.g., wait until you have N requests or a timeout, then run them all through one forward pass. This can lead to suboptimal GPU usage if some sequences finish earlier than others (the GPU might be stuck waiting for the longest sequence in the batch). Continuous batching is an improvement where the server dynamically forms batches at each step of generation. As soon as a slot in the batch is free (a sequence finished or got shorter), a new request can take its place without waiting for the entire batch to complete. The vLLM system explicitly uses this idea: it intermixes tokens from different requests in a round- robin fashion so that the GPU is never idle as long as there is any work to do 33 34. This dramatically boosts throughput under load and also keeps latency low

because new requests don't necessarily wait for an entire batch cycle - they get injected as soon as possible.

Parallelism is also needed across model parts when a single model is too large for one device. Techniques like tensor parallelism (splitting matrices across GPUs) and pipeline parallelism (each GPU handles a set of layers sequentially) were developed in training but apply to inference as well. For example, a 130B model might be split across 4 GPUs, each holding a quarter of the layers. The sequence passes through GPU1's layers, then GPU2, etc., in a pipeline. This introduces some latency overhead (communication between GPUs), but is necessary to deploy very large models. There is an art to pipeline versus batching trade- offs: with many small requests, one might prefer to replicate the model on multiple GPUs and serve different requests concurrently (data parallel inference); with one giant model and one request, one must pipeline it.

ZeRO- Inference (mentioned earlier) is a special parallelization where weights are sharded across devices and streamed in as needed 35 36 . It allows parallel fetching of different parts of a layer by multiple GPUs to speed up weight loading 37 . This kind of inference sharding is crucial when model size hits trillions of parameters.

Another straightforward parallelism is using multiple CPU cores or GPUs to handle different requests independently - essentially scaling out. But since our focus is single- model efficiency, it's worth noting that beyond a point, distributing a single model's inference has diminishing returns due to communication overhead. Many optimizations, therefore, focus on optimizing single- device performance (like better kernels, quantization, etc.), and then using parallelism primarily for scale- out or memory capacity reasons.

In sum, a well- engineered LLM serving stack will use batching to maximize hardware utilization, while balancing latency. It will continuously batch to avoid bubbles in GPU work, use caching to avoid redundant computation, and parallelize across hardware only as needed for model size or throughput. The result can be orders- of- magnitude improvements in throughput - as evidenced by vLLM's  $>20x$  throughput gain over naive approaches 17 . These algorithmic improvements don't change the model at all; they are purely about doing the same computations in a smarter order or grouping.

### 6.3 Speculative Decoding and Other Sampling Tricks

One of the most exciting recent developments in efficient inference is speculative decoding. This technique acknowledges that the biggest bottleneck in generating text is the sequential nature - the big model has to be invoked for every single token, one after the other. Speculative decoding introduces a draft model (smaller, faster) to accelerate this process 13 . Here's how it works at a high level 14 38 :

First, the small draft model is used to predict, say, the next  $Sk$  tokens in one go (it "speculates" what the large model would output). Because the draft model is, for example,  $10x$  faster, it can afford to propose multiple tokens while the big model would normally only generate one in the same time.

Then, the large target model (the original LLM) is used to verify those  $Sk$  tokens in a single parallel step. Specifically, the large model is fed the same context and asked to generate  $Sk$  outputs (or equivalently, run one forward pass on the context of length  $Sn$  to get the next token probabilities, another on context length  $Sn + 1$  for the following, etc., but this can be parallelized if reformulated cleverly). It then compares its predictions to the draft's proposal. The longest prefix of tokens where the big model agrees with the draft is accepted.

If the draft was completely correct for those  $\) k\mathbb{S}\(tokens (which happens with some probability), we just saved$ \ $k - 1\mathbb{S}$  big- model forward passes (we did one big pass that yielded \(\) k\mathbb{S}$ tokens verification). If the draft deviated at some point, we accept up to that point and then have to fallback to normal generation from the first error token onward (or some approaches regenerate from that point with the big model).

Essentially, speculative decoding uses the idle capacity of the big model's compute by feeding it multiple steps at once. The big model's forward pass is more expensive, but we do fewer of them. If the draft model has a high acceptance rate (i.e., it often predicts correctly what the big model would have done), the speedup is significant 39. OpenAI reported  $2 - 3x$  speedups with negligible difference in output quality using this method in 2023. Others have reproduced and refined it (sometimes called speculative sampling).

The critical factors are: the draft model must be well- aligned with the big model's distribution. If it's too weak and often wrong, we gain little because the big model frequently has to step in. If it's too slow (not that much faster than the big model), the benefit is also limited. Ideally, the draft is cheap and decent at prediction. Some use a smaller version of the large model (e.g. 10B draft for a 70B target), possibly distilled or fine- tuned to mimic the larger one's behavior.

This approach is interesting because it doesn't change the large model at all - it's an external augmentation. It's like a student- teacher at inference time rather than training time. And it's a rare example of breaking the strict sequential barrier of autoregressive generation without model modifications.

Besides speculative decoding, there are other sampling tricks: for instance, generating multiple tokens in parallel with the same model by generating at different beams and then conditionally accepting one - however, these often change output and are more in the realm of speeding up sampling diversity (like beam search, which is not exactly speeding up single output generation but exploring multiple). Another method is guided decoding, where a simpler model or heuristic prunes the large model's token options early, thus speeding up the softmax or sampling stage. But the gains there are minor compared to the big matrix multiplications.

We should also mention beam search and its trade- offs: beam search (keeping multiple hypotheses) can be slow for LLMs and often greedy or sampling is used instead for efficiency. Some research tries to make beam search parallel or faster, but nowadays many deployments just use greedy or nucleus sampling which are inherently one sequence at a time.

In summary, speculative decoding is a clever two- model approach that exemplifies the kind of innovation needed to push LLM inference to the next level. By sacrificing a bit of compute on a smaller model we save a lot on the bigger model. This approach and related ideas might evolve - e.g., multi- step lookahead: use a tiny model to write a whole paragraph and have the big model skim it and fix errors (a more extreme speculation). The general theme is minimize calls to the slow big model while maintaining output quality.

![](images/c9c9606958d3019a8f43532c2969b3776a0383f82915911c67d759a8ead4ed47.jpg)

Illustration of speculative decoding 14 . A small draft model proposes 3 next tokens ( "nice today outside" ). The large model in one shot evaluates those: here it agrees on "nice today" but not "outside" 40 . So those two tokens are accepted, and the large model then generates the next token after "nice today" (which is "for" ). The process repeats, allowing the draft model to accelerate generation by providing tentative tokens that the large model can quickly confirm.

Speculative decoding is particularly beneficial in latency- sensitive applications like interactive chat or real- time assistants, where a  $2 - 3x$  speedup means a noticeably snappier response 41 . It does complicate the deployment (you need two models and some coordination logic), but given the payoff, it is likely to become a standard part of the LLM serving toolbox.

## 7 System and Hardware Optimizations

At the lowest level, efficient inference is also a systems engineering challenge. This section touches on optimizations in software frameworks, hardware accelerators, and deployment strategies that complement the algorithmic techniques above.

High- Performance Kernels and Libraries: A significant part of LLM inference time is spent in linear algebra operations (matrix multiplies, laynorm, softmax). Ensuring these execute as fast as possible is crucial. Libraries like NVIDIA's TensorRT and cuBLAS, or Intel's oneDNI, provide highly optimized kernels for deep learning. Specialized routines like the aforementioned FlashAttention kernel 9 or fused ops (e.g. combined operations to reduce memory reads) can greatly speed up inference. For example, a fused kernel that does matrix multiply + bias + GELU activation in one pass saves memory bandwidth and is faster than doing them separately. Frameworks such as ONNX Runtime and PyTorch 2.0 (with TorchScript) allow exporting models to graph representations where such fusions and optimizations can be applied. Ensuring the model is run with the right batch size, using the right data layout (like channels- last, etc.), and on the right hardware units (TPUs, GPU Tensor Cores) all falls under this umbrella. In many cases, switching to an optimized runtime can yield  $2 - 3x$  speedups without any change in model accuracy 42 43 (for instance, using an int8 engine with minimal accuracy drop but significant speed gain).

Memory Offloading and CPU/GPU Coordination: Not all deployments have the latest heavy GPUs with 80GB memory. Often, models are bigger than a single accelerator's memory, or one wants to save cost by using CPU memory. Techniques like offloading swap out portions of the model or the runtime data to CPU when not needed, then swap back in when required. DeepSpeed's Zero- Inference we discussed is one extreme form: keep weights in CPU/NVMe and stream per layer 44 45. More commonly, one might offload half of the layers to a second GPU or to CPU and alternate (pipeline across heterogeneous devices). Newer memory technologies (like NVMe drives that are fast, or using slower GPU memory types) also play a role. The meta- optimizations here decide what resides where. For instance, if you have 16- bit and 8- bit versions of weights, a runtime might dynamically choose the 8- bit for speed unless it notices a drop in quality (just a conceptual possibility).

Hardware for Sparse and Low Precision: We've talked about quantization and sparsity - whether they give real speedups often depends on hardware support. NVIDIA's Amper and Hopper GPUs support 4- bit and 8- bit matrix operations and even a fixed 2:4 structured sparsity speedup (Hopper extends this further). TPUs support bfloat16 and int8. New specialized chips (from startups like Cerebras, Groq, Mythic, etc.) claim to run transformers more efficiently, some even storing weights in analog form or leveraging massive sparsity. One noteworthy trend is FP8 (8- bit float) which is being standardized; it could combine the range of floating point with the compactness of 8 bits, and hardware like NVIDIA H100 supports it. If models can be quantized to FP8 with no loss, inference would both speed up and simplify (no need for complex calibration as with int8).

Energy and Cooling Considerations: Efficiency is not just speed - it's also doing more with less energy. A method that doubles throughput on the same hardware implicitly halves the energy per inference. This is important for environmental and cost reasons. Some hardware features like DVFS (dynamic frequency scaling) can downclock when utilization is low - ironically, inefficient code might use less of the GPU and then not get full performance per watt. So maximizing utilization also often maximizes performance per watt. In settings like mobile devices, using int8 or even int4 can be the difference between running in realtime or not at all, due to limited battery and thermal headroom. Qualcomm, Apple, and other mobile chipset companies have added specific accelerator blocks for transformer and int8 operations to enable on- device LLM inference.

Scalability and Distribution: When serving at large scale, efficiency also means horizontal scalability - e.g., load balancing queries across many machines, dynamically scaling up/down model instances. A well- optimized inference service might spin up multiple model replicas and use a routing layer to ensure each GPU is busy. This goes a bit beyond the scope of model- centric optimizations, but it's the real- world aspect: an efficient model plus an inefficient serving architecture can still underperform. Thus, research like Alpa and Ray Serve looks at automatically distributing model inference across clusters for both speed and fault tolerance.

Frameworks and Tooling: There are now numerous toolkits dedicated to LLM inference: besides vLLM, there is Hugging Face's Text Generation Inference (TGI) server, Triton Inference Server by NVIDIA, DS- Inference by DeepSpeed, and others from Alibaba, ByteDance etc., each implementing many of the techniques we've described (quantization, parallelism, caching, batching). These tools make it easier for practitioners to get efficiency without reinventing wheels. The existence of ML commons benchmarks (like MLPerf) for inference is also driving standard practices - e.g. it's now expected to use int8 and batching to hit good scores on BERT or GPT inference benchmarks.

In summary, system- level and hardware optimizations form the backbone that enables all the model- centric innovations to fully manifest as wall- clock speedups. A 4- bit quantized model needs a good kernel to run fast; a pruned model needs sparse matrix libraries to see gains; a draft- model approach needs multi- model serving support. The co- design of models with hardware is an emerging theme - for

instance, maybe future LLMs will incorporate blocks that are intentionally easily quantizable or have sparse structures tuned to the next generation of AI chips. The ultimate vision is to be able to run a dialogue with a powerful LLM on a consumer phone or to serve a billion queries a day on a reasonable GPU cluster – goals that would have seemed outlandish a few years ago, but are getting closer due to these efficiency advances.

## 8 Applications and Impact Across Domains

Techniques for efficient LLM inference are enabling a range of applications that previously would be infeasible or too costly. We highlight a few areas where these advancements have broad impact:

- Edge and On-Device AI: Perhaps the most visible benefit is the ability to run language models on personal devices (phones, laptops) or edge servers. Quantization and distillation have allowed models like 7B-parameter LLaMA variants to be squeezed onto smartphones (with 4-bit weights) and still perform decently for tasks like drafting messages or answering queries offline. This opens up privacy-preserving applications (since data need not be sent to a server) and personalization (the model on your device can be fine-tuned to your data). Efficient inference is the key – a device with only a few watts of power can’t run a 175B model at FP16 in realtime, but it might run a 4-bit 7B model that was distilled, achieving useful functionality. For instance, an on-device translation or keyboard assistant model can now be as powerful as cloud models from a few years ago, thanks to compression.

- Real-Time Communication: In domains like live translation, subtitles generation, or assistive technology (e.g. an AR device whispering answers to you), latency is critical. A delay of more than say 200 milliseconds can be noticeable. By applying speeding-up techniques (caching, batching, even speculative decoding if applicable), LLMs can start to meet these real-time constraints. For example, a meeting translation app might use an efficient long-text model that can process what’s being said on the fly without a massive GPU farm – benefitting from efficient attention and quantization to run on a single GPU with low latency.

- Interactive Education and Companions: There is growing interest in AI tutors, companions, and interactive fiction. These require both good conversational quality (often needing a large model) and fast response to maintain a natural interaction. Efficient inference makes it economically feasible to deploy such assistants widely (e.g. in classrooms or as personal tutor apps). If each instance can run on cheaper hardware, schools or users can have their own local AI without relying on expensive cloud usage. Moreover, frameworks like continuous batching help in serving many users simultaneously from one model instance, which is crucial for scalability in these applications.

- Enterprise Applications and APIs: Companies providing LLM-powered services (from copywriting to code generation) have strong financial incentives to optimize inference. Serving a large model in the cloud costs significant GPU time and energy. Techniques like model pruning and distillation allow them to deploy smaller versions fine-tuned for their specific domain (legal, medical, programming etc.), which cut costs. Speculative decoding and multi-model setups allow them to use fewer resources per query. This translates to cheaper or even on-premise solutions for enterprise clients who might deploy the models on their own servers with limited GPU capacity.

- Multimodal and Hybrid Models: Efficient inference methods aren’t limited to pure text models. Multimodal models (like GPT-4 vision, or image captioning models) also benefit. For instance,

stable diffusion image generation models have seen huge speedups via quantization (running on CPUs in seconds instead of minutes) and optimizations like half- precision. Similarly, large speech recognition models or video understanding models apply these same principles. There's cross- pollination: techniques developed for vision (like pruning CNNs) inspired analogous ideas in LLMs, and vice versa (transformer accelerators built for text can help in speech or video tasks).

- Research and Prototyping: When models run faster, researchers can iterate quicker and test ideas that require many model calls. For example, doing reinforcement learning with human feedback or automated search with an LLM in the loop is vastly more tractable if each inference is cheap. Efficient inference thus accelerates research progress itself. It also democratizes it – more academic labs and even hobbyists can experiment with LLMs if they can compress them to run on accessible hardware.

In all these cases, the common theme is: efficiency turns what was once impossible or costly into the new normal. A historical analogy is how compression and optimized hardware made it feasible to put a speech recognizer on a phone – something that initially required server farms. We are witnessing a similar transition for large language models.

## 9 Open Challenges and Future Directions

Despite impressive progress, several challenges remain open in making LLM inference truly efficient and scalable:

- Maintaining Accuracy in Extreme Compression: Pushing quantization below 4 bits, or pruning beyond  $90\%$  sparsity, or distilling to extremely tiny models (like  $< 1B$  parameters) often results in noticeable performance drops. Bridging this gap is an ongoing area of research. Techniques like quantization-aware training (fine-tuning the model with quantization in the loop) can help regain some accuracy  $46 - 47$ , but at significant training cost. Future research might find better theoretical understanding of model redundancy to guide compression without loss. Also, lossless compression in a sense: clever weight coding or sharing that reduces memory without changing the computation could be explored (e.g. using one weight matrix to serve multiple purposes).

- Hardware Utilization for Sparse Models: Unstructured sparsity (from pruning) has not delivered proportional speedups on GPUs yet because current hardware is not optimized for it. This presents both a challenge and an opportunity: either new hardware needs to incorporate efficient sparse computation (some AI chips are doing this), or algorithms might need to impose more structured sparsity that hardware can use (e.g. block-sparsity patterns). There's also the dynamic sparsity of MoES – currently, the overhead of gating and distributed memory for experts can reduce gains. Research into load balancing algorithms for MoE inference and better routing methods could make large MoEs more practically deployable.

- Scaling to Longer Contexts: While efficient attention methods exist, using extremely long contexts (like tens or hundreds of thousands of tokens) still often requires either massive compute or approximations that might miss some dependencies. Approaches like hierarchical models (first summarize or chunk the input, then reason) or retrieval instead of long context have been proposed to avoid brute-force long attention. An open question is: what is the best way for an LLM to handle a book-length input efficiently? Possibly through a mix of retrieval (to not read everything at once) and memory (to not forget earlier parts). Making such processes as seamless and general as the original transformer is a challenge.

- Real-Time and Streaming Outputs: For tasks like simultaneous translation or running commentary, the model must generate text concurrently with receiving input. This requires careful pipeline designs and possibly modifications to the model (e.g. a unidirectional mode that can be paused and resumed). Ensuring ultra-low latency (few milliseconds per token) will be crucial in these streaming scenarios. It might involve further specialization of models or the use of approximate caching (like selectively updating only parts of the output as new info comes in).

- Energy Efficiency and Green AI: Even with improvements, large models consume significant energy. Techniques purely focusing on speed might inadvertently increase energy (e.g., speculative decoding uses extra compute on the draft model). The future goal is to make models not just faster but more energy-proportional. This might involve new metrics for efficiency that weigh quality vs energy. There's interest in analog computing or non-digital accelerators that could run models with lower power, but those come with their own challenges in precision and reliability.

- Robustness of Compressed Models: Sometimes compression can affect a model's outputs in subtle ways 
- e.g. quantized models might be more prone to certain errors or biases because of the reduced precision. Ensuring that optimized models remain safe and aligned (especially important for chatbots that could output harmful content) is an open area. Compression-aware evaluation on safety and bias metrics is something that will likely become standard.

- Automated Co-Design: Right now, engineers manually combine quantization, distillation, etc., for each model. A future direction is automated tools that, given a trained large model and deployment constraints, can automatically produce an optimized model and runtime. This might use neural architecture search to decide which layers to prune or which parts to quantize more aggressively, yielding a custom optimized model for a specific use-case. Early steps in this direction are appearing (some academic works on automatically mixed precision assignments, for instance).

- Balancing Versatility with Efficiency: Some efficient strategies involve specializing the model (like distilling for a specific task or domain). But we also value the general-purpose nature of foundation models. An ongoing challenge is how to compress models while keeping them generalists. One potential direction is multi-model systems where a moderate core model handles most and calls a specialized large model only when needed (sort of an on-demand scaling). This relates to ideas of modular AI where different components (some small, some large) are orchestrated efficiently.

In essence, making LLMs efficient is a multi- faceted problem spanning algorithmic, systems, and even theoretical domains. Each gain in efficiency often exposes another bottleneck to tackle (we made computation faster, now memory bandwidth is the limiter; we pruned weights, now irregular memory access is an issue; etc.). The community is actively addressing these, and given the tremendous incentives (cost reduction, enabling new applications, democratizing AI), we can expect rapid progress.

## 10 Conclusion

Efficient inference for large language models is transforming the landscape of what is possible with AI. Where once only a handful of tech giants could afford to deploy massive models, today we see optimized LLMs running on laptops and phones, and being integrated into a myriad of products and services. This survey reviewed the key concepts, methods, and developments enabling this shift: from compressing model size via quantization, distillation, and pruning, to re- thinking model architectures for sparsity and

adaptivity, to optimizing the inference process through caching, batching, and novel algorithms like speculative decoding. We also touched on the systems- level innovations and hardware trends that complement these techniques.

A few themes emerge. Redundancy in large models - LLMs have more capacity than needed for any single task or token, and efficiency techniques aim to leverage that redundancy, either by removing it (compression) or by bypassing it when not needed (sparse/conditional computation). Another theme is alignment between model and hardware - the closer the model's operations match what hardware can do quickly (e.g., matrix multiplies on low precision, or skipping zeros), the faster it runs. We see a coevolution: as models push certain patterns (Transformer has lots of matrix ops), hardware adapts (GPUs optimize dense GEMMs); now models are pushing sparsity and hardware is beginning to follow.

Crucially, efficient inference is not just about speed for speed's sake - it unlocks AI deployment in the real world. It enables responsiveness in user interactions, reduces the carbon footprint of AI computations, and lowers barriers to entry so that innovation isn't limited to those with vast compute resources. A well- optimized large model can truly be a point of leverage, giving perhaps  $10x$  the user benefit at  $1x$  the cost of an unoptimized approach.

Looking ahead, the gap between training and inference might narrow: techniques like quantization- aware training or sparsity- aware training will bake efficiency into the model from the start. We might also witness new model paradigms explicitly designed for efficiency, moving beyond the Transformer but retaining its strengths - for instance, models that can reason in a modular way or use external tools (so they don't have to "think" as hard internally). The continued growth in model scale will always pose new challenges (e.g., if we have a 10 trillion parameter model, how to inference that efficiently will be a new frontier), but the lessons learned so far will guide solutions for the next generation.

In conclusion, efficient LLM inference is a vibrant and interdisciplinary field. By combining insights from machine learning, software engineering, and hardware design, we are making the once- impractical feasible. This survey provided an overview of the state- of- the- art as of 2025. As the field progresses, we expect even more creative solutions to emerge, ultimately moving us closer to the ideal of "anyone can use an advanced AI anytime, anywhere" without worrying about the computations under the hood.

Acknowledgements: This survey drew upon numerous sources and research works cited throughout, reflecting the contributions of the community at large in advancing efficient AI. We encourage readers to explore those references (and the many more we couldn't cover) for a deeper dive into specific topics. Efficient inference is a quickly evolving area, and staying updated through arXiv papers, blogs, and system benchmarks will be key for practitioners and researchers alike.

## References

1 13 14 38 39 40 41 Get  $3x$  Faster LLM Inference with Speculative Decoding Using the Right Draft Model

https://www.bentoml.com/blog/3x- faster- llm- inference- with- speculative- decoding

2 3 Large language model - Wikipedia https://en.wikipedia.org/wiki/Large_language_model

4 Introduction to DistilBERT in Student Model - Analytics Vidhya https://www.analyticsvidhya.com/blog/2022/11/introduction- to- distilbert- in- student- model/

5 21 GPT3. int8(): 8- bit Matrix Multiplication for Transformers at Scale https://openreview.net/forum?id=dXiGWqBoxaD

6 7 27 28 Mixture of Experts Explained https://huggingface.co/blog/moe

8 35 36 37 44 45 ZeRO- inference: Democratizing massive model inference - DeepSpeed https://www.deepspeed.ai/2022/09/09/zero- inference.html

9 FlashAttention: Fast and Memory- Efficient Exact Attention with IO ... https://arxiv.org/abs/2205.14135

10 12 20 46 47 A Comprehensive Evaluation of Quantization Strategies for Large Language Models https://arxiv.org/html/2402.16775v1

11 22 23 24 A Visual Guide to Quantization - Maarten Grootendorst https://www.maartengrootendorst.com/blog/quantization/

15 16 17 31 32 33 34 42 43 Meet vLLM: For faster, more efficient LLM inference and serving https://www.redhat.com/en/blog/meet- vllm- faster- more- efficient- llm- inference- and- serving

18 Mixtral 8x7B: a new MLPerf Inference benchmark for mixture of experts https://mlcommons.org/2024/08/moe- mlperf- inference- benchmark/

19 LLM.int8() and Emergent Features - Tim Dettmers https://timdettmers.com/2022/08/17/llm- int8- and- emergent- features/

25 Massive Language Models Can Be Accurately Pruned in One- Shot https://arxiv.org/abs/2301.00774

26 locuslab/wanda: A simple and effective LLM pruning approach. https://github.com/locuslab/wanda

29 Toward Inference- optimal Mixture- of- Expert Large Language Models https://arxiv.org/abs/2404.02852

30 Why large language models struggle with long contexts https://www.understandingai.org/p/why- large- language- models- struggle