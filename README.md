The ever-increasing demands for intuitive interactions in virtual reality have triggered a boom in facial expression recognition (FER). To address the limitations in existing approaches (e.g., narrow receptive fields and homogenous supervisory signals) and further cement the capacity of FER tools, a novel multimodal supervision-steering transformer for FER in the wild is proposed in this paper. Referring to the FER-former, our approach features multigranularity embedding integration, a hybrid self-attention scheme, and heterogeneous domain-steering supervision. Specifically, to dig deep into the merits of the combination of features provided by prevailing CNNs and transformers, a hybrid stem is designed to cascade two types of learning paradigms simultaneously. A FER-specific transformer mechanism is devised to characterize conventional hard one-hot label-focusing and CLIP-based text-oriented tokens in parallel for final classification. To ease the issue of annotation ambiguity, a heterogeneous domain-steering supervision module is proposed to add text-space semantic correlations to image features by supervising the similarity between image and text features.
On top of the collaboration of multifarious token heads, diverse global receptive fields with multimodal semantic cues are captured, delivering superb learning capability. Extensive experiments on popular benchmarks demonstrate the superiority of the proposed FER-former over the existing state-of-the-art methods.