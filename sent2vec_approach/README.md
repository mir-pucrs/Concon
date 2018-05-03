# Conflict Identification using Sent2Vec

    In this approach, we use sentence embeddings to compare norm embeddings and identify conflicts between them.
    We use sentence embeddings to represent textual sentences into latent vectors that maintain syntactic and semantic information.
    In order to identify conflicts between norms, we generate an offset vector containing the conflict meaning.
    By conflict meaning, we mean that we generate a latent vector containing values that represent conflicts between norms.
    Using such offset, we can compare it to the difference between two norm embeddings and, given a threshold, indicate whether both norms are conflicting or not.

### Sent2Vec

    In order to generate the embeddings for norm sentences, we use the sent2vec proposed by Pagliardini et al. and better described in their [repository](https://github.com/epfml/sent2vec)


### Offset Creation

    In order to create an offset containing the norm conflict meaning, first we need a set of conflicting norm pairs.
    As dataset, we use our manually generated conflicts (available [here](https://zenodo.org/record/345411#.WutwDnUvxhF)).
    The process consists of 


### TODO: Add our accepted paper on IJCNN 2018