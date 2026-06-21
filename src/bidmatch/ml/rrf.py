from collections import defaultdict


def reciprocal_rank_fusion(
        rankings: list[list[int]],
        weights: list[float] = None,
        k: int = 60
) -> list[tuple[int, float]]:
    """
    Két vagy több ranglista összefésülése RRF algoritmussal.
 
    RRF score = Σ weight_i / (k + rank_i)
 
    k=60 az irodalomban bevált alapértelmezett (Cormack et al. 2009).
    Magasabb k → egyenletesebb súlyozás; alacsonyabb k → top helyezések jobban számítanak.
 
    Args:
        rankings : lista ranglistákból, ahol minden ranglista corpus indexeket tartalmaz
                   relevancia sorrendben (legjobb az első)
        weights  : opcionális súlyok (alapértelmezett: egyenlő súly)
        k        : RRF konstans
 
    Returns:
        Lista (corpus_index, rrf_score) párokból, csökkenő sorrendben
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    
    scores: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_idx in enumerate(ranking):
            scores[doc_idx] += weight / (k + rank + 1)
 
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)