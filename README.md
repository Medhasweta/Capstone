# Data Science Program Capstone Report

This project delves into enhancing Question Answering systems using Knowledge Graphs (KGs). By integrating these graphs with advanced machine learning models, we aim to improve the accuracy and efficiency of systems that answer complex questions.

### Authors
- Medhasweta Sen
- Nina Ebensperger
- Nayaeun Kwon

### Supervised By
- Dr. Amir Jafari

---

## Project Overview

**Abstract**: This project investigates the impact of negative sampling on Knowledge Graph Embeddings (KGEs) for Knowledge Graph Question Answering (KGQA) systems. We explore various negative sampling techniques and KGE models like DistMult and ComplEx, assessing their effects across several knowledge graph benchmarks.

### Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Knowledge Graphs and Knowledge Graph Embeddings](#knowledge-graphs-and-knowledge-embeddings)
- [EmbedKGQA: The Question Answering Pipeline](#embedkgqa-the-question-answering-pipeline)
- [Experiment](#experiment)
- [Results and Discussion](#results-and-discussion)
- [Conclusion and Future Research Directions](#conclusion-and-future-research-directions)

---

## Introduction

**Challenges Addressed**:
- **Complex Queries**: Enhancing the ability of systems to understand and process complex queries by leveraging the structured relationships stored in Knowledge Graphs.
- **Sparse Data**: Improving system robustness in handling sparse data areas within databases, which are typically challenging for traditional methods.


![General Framework of LLMs for KGQA](https://miro.medium.com/v2/resize:fit:1086/format:webp/1*JSHShzlkC_pAMpI-U4VKIg.png)


---

## Related Work

The development of Knowledge Graph Question Answering (KGQA) systems has significantly advanced with the integration of Knowledge Graph Embeddings (KGEs) and Large Language Models (LLMs). This section reviews notable methodologies and models that have shaped the current landscape of KGQA, particularly focusing on their application in enhancing the effectiveness and accuracy of question answering over knowledge graphs.

### Knowledge Graph Embeddings Computation

#### ComplEx
Introduced by Trouillon et al. (2016), the ComplEx model employs complex-valued embeddings to handle both symmetric and antisymmetric relations effectively within knowledge graphs. This model is renowned for its use of complex numbers to enrich the representation of entities and relations, thereby allowing for the effective capture of intricate relational patterns. The ComplEx model is distinguished by its use of the Hermitian dot product to calculate the interaction between embeddings, offering a robust method for link prediction that has shown superior performance on benchmarks like FB15K and WN18 ([Trouillon et al., 2016](https://arxiv.org/abs/1606.06357)).

#### DistMult
The DistMult model from Yang et al. (2015) simplifies the embedding process by utilizing real-valued vectors to represent entities and relations, making it highly efficient for large-scale knowledge bases. DistMult applies a diagonal matrix to model relations within its bilinear scoring function, which has been effective in tasks such as link prediction and rule mining. This model's ability to generalize several existing embedding models highlights its utility in enhancing scalability and interpretability within KGQA systems ([Yang et al., 2015](https://arxiv.org/abs/1412.6575)).

### Question Answering over Knowledge Graphs

The evolution of KGQA has been marked by a shift from basic embedding models to more sophisticated, multi-dimensional approaches that accommodate complex, multi-hop queries. Early systems relied on models like TransE, which, while effective in simplifying entities and relations, often struggled with the complexity of multi-hop question answering due to their limited relational path encapsulation capabilities.

Significant enhancements in KGQA include the integration of complex vector space embeddings like ComplEx, which enable dynamic interpretation and linking of relational entities. Further advancements have incorporated methods for extracting sub-graphs from knowledge graphs, reducing computational overhead and improving accuracy by focusing on relevant graph sections.

#### Embed-KGQA Model
The Embed-KGQA model, as detailed by Saxena et al. (2020), represents a synthesis of these advancements, incorporating complex embeddings and neural techniques to enhance question answering capabilities. This model employs a structured pipeline consisting of a KG Embedding Module, a Question Embedding Module, and an Answer Selection Module, each designed to optimize the accuracy and efficiency of KGQA systems on extensive knowledge bases.

### Future Directions
While current advancements have significantly improved KGQA systems, challenges in scalability and adaptability remain. Future research may explore more adaptive models that update embeddings in real-time and integrate more detailed linguistic features to refine accuracy further.

For detailed insights and references, please consult the works of [Trouillon et al., 2016](https://arxiv.org/abs/1606.06357), [Yang et al., 2015](https://arxiv.org/abs/1412.6575), and [Saxena et al., 2020](https://aclanthology.org/2020.acl-main.412).

---

## Knowledge Graphs and Knowledge Graph Embeddings

Knowledge Graphs (KGs) play a pivotal role in structuring and harnessing complex data across diverse domains. Defined as a collection of triples, KG = {(h, r, t) ⊆ E × R × E}, where E represents entities and R denotes relations, these graphs store knowledge in a structured, semantically rich format. This structure not only supports advanced data integration and querying but also enhances analysis capabilities crucial for intelligent systems.

### Types of Knowledge Graphs
Knowledge graphs can be categorized based on the nature of the stored information:
- **Encyclopedic KGs:** Such as Wikidata, Freebase, and Dbpedia, which provide broad informational coverage by compiling general knowledge from extensive sources.
- **Commonsense KGs:** Like ConceptNet and ATOMIC, encode everyday knowledge about objects and interactions, aiding machines in understanding human concepts.
- **Domain-Specific KGs:** Focus on specialized fields like medicine or finance, emphasizing precision and relevance over breadth.
- **Multimodal KGs:** Such as IMGpedia and MMKG, integrate data across various modalities—text, images, videos—to enhance data richness and support complex applications like visual question answering.

### Importance of Knowledge Graph Embeddings (KGEs)
While KGs are effective for direct queries, their structural form can limit applications requiring inferential reasoning, such as link prediction or similarity assessment. KGEs transform these entities and relationships into continuous vector spaces, enabling the use of mathematical tools to measure similarities and predict relationships. This transformation is crucial for tasks like recommendation systems or social network analysis and aids in integrating KGs with other machine learning models for enhanced NLP system performance.

### Leveraging Negative Sampling Techniques for KGEs
Negative sampling is essential in KGE training, helping models learn to distinguish between true and potential but unobserved relationships. This process involves:
- **Discriminative Learning:** Learning to preferentially score true relationships higher than false ones.
- **Handling Sparse Data:** Addressing the sparsity of KGs by teaching models about potential non-existing relations.
- **Improving Generalization:** Encouraging models to generalize well by providing a variety of counterexamples.

**Static Random Negative Sampling**, a prevalent approach, applies the closed-world assumption, considering any triple not present in the KG as negative. However, this method faces challenges due to the inherent incompleteness of KGs and might generate incorrect negative instances, leading to errors in model training.

For a deeper understanding, refer to the works of [Pan et al., 2023](https://example.com/pan2023), [Madushanka and Ichise, 2024](https://example.com/madushanka2024), and other leading researchers in the field.

![Examples of KG Categories](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*fKqdqCMI9UQz0YX291IVIw.png)

---

## EmbedKGQA: The Question Answering Pipeline

EmbedKGQA methodically answers complex questions using KGs through its three main modules: KG Embedding, Question Embedding, and Answer Selection. This pipeline optimizes the question answering process with advanced embedding techniques.

![Overview of EmbedKGQA](link-to-image)

---

## Experiment

### Dataset

We utilize the MetaQA dataset, a multi-hop KGQA dataset focused on the movie domain, to evaluate our KGQA system. This dataset includes extensive queries connected to a comprehensive movie knowledge graph.

### Benchmarking with Classical Method: Cosine Similarity

This method measures the effectiveness of the EmbedKGQA system by comparing it to traditional cosine similarity approaches in text analysis.

### Research Design

The experiment investigates various negative sampling techniques and their impact on KGEs by employing models like ComplEx and DistMult and assessing their performance through different metrics.

---

## Results and Discussion

### Performance Evaluation of Knowledge Graph Embeddings (KGEs)

The effectiveness of different KGE models under various negative sampling techniques was assessed using three key performance metrics:

- **Mean Rank**: Measures the average position of the correct entity in a list ranked by the model. A lower mean rank indicates better performance, as the correct entities are ranked closer to the top ([Bordes et al., 2013](https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)).
- **Mean Reciprocal Rank (MRR)**: This metric evaluates the average inverse rank of the correct entities, with higher values indicating superior model performance in ranking correct entities near the top of the list ([Yang et al., 2015](https://arxiv.org/abs/1412.6575)).
- **Hits@k**: Measures the percentage of correct entities that appear within the top k ranks, with higher values suggesting better retrieval performance ([Yang et al., 2015](https://arxiv.org/abs/1412.6575)).

The comparative analysis revealed that ComplEx, particularly with Random Corrupt Negative Sampling, outperformed other configurations, including DistMult. The table below summarizes the key metrics for different sampling and embedding methods:

| KGE Model  | Sampling Method     | Mean Rank | MRR    | Hits@10 |
|------------|---------------------|-----------|--------|---------|
| ComplEx    | Uniform             | 397.48    | 0.2742 | 0.3664  |
| ComplEx    | Random Corrupt      | 357.62    | 0.2849 | 0.3886  |
| ComplEx    | Batch NS            | 463.07    | 0.2668 | 0.3459  |
| DistMult   | Uniform             | 528.17    | 0.2386 | 0.3360  |
| DistMult   | Random Corrupt      | 477.78    | 0.2554 | 0.3400  |
| DistMult   | Batch NS            | 660.01    | 0.1922 | 0.2756  |

### Insights and Implications

The superior performance of ComplEx can be attributed to its ability to effectively model both symmetric and antisymmetric relations, thanks to its complex-valued embeddings. These embeddings enhance the model’s capability to distinguish between different types of relational structures, a critical factor in accurately processing complex KGQA tasks.

DistMult’s limitations stem from its symmetric scoring function, which cannot differentiate well between relational structures where directionality is crucial, such as antisymmetric relations. This fundamental difference in handling relation types is a key factor in the observed performance disparity between ComplEx and DistMult.

The findings from this study not only highlight the importance of choosing the right negative sampling technique but also underscore the potential of complex embeddings in enhancing the performance of KGQA systems.

For a more detailed exploration of these results and methodologies, refer to the studies by [Trouillon et al., 2016](https://arxiv.org/abs/1606.06357) and [Yang et al., 2015](https://arxiv.org/abs/1412.6575).


---

## Conclusion and Future Research Directions

This study highlights the pivotal role of negative sampling in enhancing the performance of KGQA systems and suggests future research areas such as dynamic adaptation of embeddings and the integration of multimodal data.

---

**Video Demonstration:** [Click here to watch]([link-to-video](https://drive.google.com/file/d/1iGQIY6rx7LpiDxoI0BDY2RzK03k5vArc/view?usp=sharing))

**Dataset:** [Access the dataset here](https://drive.google.com/drive/folders/1IVUUoo_cvmLJBWSl4BOln8YL4VoMC9LM?usp=sharing)

**References:**
1. Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2023). Unifying Large Language Models and Knowledge Graphs: A Roadmap. *arXiv preprint arXiv:2306.08302*. [Access here](https://api.semanticscholar.org/CorpusID:259165563).
2. Saxena, A., Tripathi, A., & Talukdar, P. (2020). Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*. [Access here](https://aclanthology.org/2020.acl-main.412).
3. Rao, D. J., Mane, S. S., & Paliwal, M. A. (2022). Biomedical Multi-hop Question Answering Using Knowledge Graph Embeddings and Language Models. *arXiv preprint arXiv:2211.05351*.
4. Yang, B., Yih, W., He, X., Gao, J., & Deng, L. (2015). Embedding Entities and Relations for Learning and Inference in Knowledge Bases. *arXiv preprint arXiv:1412.6575*.
5. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *Neural Information Processing Systems*.
6. Sun, Z., Deng, Z., Nie, J., & Tang, J. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *arXiv preprint arXiv:1902.10197*.
7. Yang, M., Lee, D., Park, S., & Rim, H. (2015). Knowledge-based Question Answering Using the Semantic Embedding Space. *Expert Systems with Applications*, 42, 9086-9104. [DOI](10.1016/j.eswa.2015.07.009).
8. Bao, J., Duan, N., Yan, Z., Zhou, M., & Zhao, T. (2016). Constraint-Based Question Answering with Knowledge Graph. *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers*, 2503–2514. [Access here](https://aclanthology.org/C16-1236).
9. Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex Embeddings for Simple Link Prediction. *arXiv preprint arXiv:1606.06357*.
10. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *Advances in Neural Information Processing Systems*.
11. Sun, H., Dhingra, B., Wang, M., Mazaitis, K., Salakhutdinov, R., & Cohen, W. W. (2018). Open Domain Question Answering Using Early Fusion of Knowledge Bases and Text. *arXiv preprint arXiv:1809.00782*.
12. Sun, H., Bedrax-Weiss, T., & Cohen, W. W. (2019). PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text. *arXiv preprint arXiv:1904.09537*.
13. Madushanka, T., & Ichise, R. (2024). Negative Sampling in Knowledge Graph Representation Learning: A Review. *arXiv preprint arXiv:2402.19195*.
14. Ganesan, B., Ravikumar, A., Piplani, L., Bhaumurk, R., Padmanaban, D., Narasimhamurthy, S., Adhikary, C., & Deshapogu, S. (2024). Automated Answer Validation using Text Similarity. *arXiv preprint arXiv:2401.08688*.
15. Yih, W., Chang, M., He, X., & Gao, J. (2015). Semantic Parsing for Single-Relation Question Answering. *Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics*, 643–648. [Access here](https://aclanthology.org/).
16. Zhang, Y., Dai, H., Kozareva, Z., Smola, A., & Song, L. (2017). Variational Reasoning for Question Answering With Knowledge Graph. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).
17. Murtagh, F., & Legendre, P. (2014). Ward’s Hierarchical Agglomerative Clustering Method: Which Algorithms Implement Ward’s Criterion? *Journal of Classification*, 31(3), 274–295. [DOI](10.1007/s00357-014-9161-z).

---

For any inquiries or further information, please contact [Our Wonderful Project](mailto:contact@ourwonderfulproject.org).

---
