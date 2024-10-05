import numpy as np
from sklearn.preprocessing import OneHotEncoder

from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE
from torchrbm.dataset.fasta_utils import compute_weights
from torchrbm.dataset.fasta_utils import encode_sequence
from torchrbm.dataset.fasta_utils import get_tokens
from torchrbm.dataset.fasta_utils import import_from_fasta
from torchrbm.dataset.fasta_utils import validate_alphabet


def load_FASTA(
    filename,
    variable_type="Potts",
    use_weights: bool = False,
    alphabet: str = "protein",
    device="cuda"
):
    # Select the proper encoding
    tokens = get_tokens(alphabet)
    names, sequences = import_from_fasta(filename)
    validate_alphabet(sequences=sequences, tokens=tokens)
    names = np.array(names)
    dataset = np.vectorize(
        encode_sequence, excluded=["tokens"], signature="(), () -> (n)"
    )(sequences, tokens)

    num_data = len(dataset)
    if use_weights:
        print("Automatically computing the sequence weights...")
        weights = compute_weights(dataset, device=device)
    else:
        weights = np.ones((num_data, 1), dtype=np.float32)

    weights = weights.squeeze(-1)
    match variable_type:
        case "Potts":
            pass
        case "Bernoulli":
            num_categories = len(np.unique(dataset))
            num_visibles = dataset.shape[1]
            categories = (
                np.arange(num_categories)
                .repeat(num_visibles, axis=0)
                .reshape(-1, num_visibles)
                .T
            )
            enc = OneHotEncoder(categories=categories.tolist())
            dataset = enc.fit_transform(dataset).toarray()
        case "Ising":
            num_categories = len(np.unique(dataset))
            num_visibles = dataset.shape[1]
            categories = (
                np.arange(num_categories)
                .repeat(num_visibles, axis=0)
                .reshape(-1, num_visibles)
                .T
            )
            enc = OneHotEncoder(categories=categories.tolist())
            dataset = enc.fit_transform(dataset).toarray() * 2 - 1
    return dataset, weights, names


def load_BKACE(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/BKACE.fasta",
    variable_type="Potts",
    use_weights=False,
    alphabet="protein",
):
    return load_FASTA(
        filename=filename,
        variable_type=variable_type,
        use_weights=use_weights,
        alphabet=alphabet,
    )


def load_PF00072(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/PF00072.fasta",
    variable_type="Potts",
    use_weights=False,
    alphabet="protein",
):
    return load_FASTA(
        filename=filename,
        variable_type=variable_type,
        use_weights=use_weights,
        alphabet=alphabet,
    )


def load_PF13354(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/PF13354.wo_ref_seqs.fasta",
    variable_type="Potts",
    use_weights=False,
    alphabet="protein",
):
    return load_FASTA(
        filename=filename,
        variable_type=variable_type,
        use_weights=use_weights,
        alphabet=alphabet,
    )
