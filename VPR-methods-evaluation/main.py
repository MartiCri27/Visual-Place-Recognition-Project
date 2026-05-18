import parser
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import faiss
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import visualizations
import vpr_models
from test_dataset import TestDataset

import faiss

def main(args):
    # Registra l'ora di inizio dell'esecuzione
    start_time = datetime.now()

    # Rimuove i logger precedenti e configura i nuovi logger
    logger.remove()  # Rimuove i logger precedentemente esistenti
    log_dir = Path("logs") / args.log_dir / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    logger.add(log_dir / "debug.log", level="DEBUG")
    logger.info(" ".join(sys.argv))
    logger.info(f"Arguments: {args}")
    logger.info(
        f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    )
    logger.info(f"The outputs are being saved in {log_dir}")

    # Carica il modello e lo configura per la modalità di valutazione
    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    # Carica il dataset di test con immagini di database e query
    test_ds = TestDataset(
        args.database_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    logger.info(f"Testing on {test_ds}")

    # Estrae i descrittori dalle immagini senza calcolare i gradienti
    with torch.inference_mode():
        logger.debug("Extracting database descriptors for evaluation/testing")
        # Estrae i descrittori del database
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        # Estrae i descrittori delle query con batch size 1
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    # Separa i descrittori delle query da quelli del database
    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    # Salva i descrittori su disco se richiesto
    if args.save_descriptors:
        logger.info(f"Saving the descriptors in {log_dir}")
        np.save(log_dir / "queries_descriptors.npy", queries_descriptors)
        np.save(log_dir / "database_descriptors.npy", database_descriptors)


    #Modifica per supportare sia L2 che inner product come metrica per la ricerca kNN
    # Crea un indice FAISS per la ricerca dei vicini più prossimi (kNN)
    #faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    #faiss_index.add(database_descriptors)
    #del database_descriptors, all_descriptors

    if args.knn_metric == "l2":
        faiss_index = faiss.IndexFlatL2(args.descriptors_dimension)
    elif args.knn_metric == "ip":
        faiss_index = faiss.IndexFlatIP(args.descriptors_dimension)
    else:
        raise ValueError(f"Unknown metric: {args.knn_metric}")
    
    faiss_index.add(database_descriptors)

    del database_descriptors, all_descriptors

    # Calcola le distanze e le predizioni mediante ricerca kNN
    logger.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_descriptors, max(args.recall_values))

    # Calcola i recall se le etichette sono disponibili
    if args.use_labels:
        positives_per_query = test_ds.get_positives()
        recalls = np.zeros(len(args.recall_values))
        # Per ogni query, verifica se le predizioni corrette sono nei top-n
        for query_index, preds in enumerate(predictions):
            for i, n in enumerate(args.recall_values):
                if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                    recalls[i:] += 1
                    break

        # Converte i recall in percentuali
        recalls = recalls / test_ds.num_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
        logger.info(recalls_str)

    # Salva le visualizzazioni delle predizioni
    if args.num_preds_to_save != 0:
        logger.info("Saving final predictions")
        # Per ogni query salva le prime num_preds_to_save predizioni
        visualizations.save_preds(
            predictions[:, : args.num_preds_to_save], test_ds, log_dir, args.save_only_wrong_preds, args.use_labels
        )

    # Salva i dati per l'analisi dell'incertezza se richiesto
    if args.save_for_uncertainty:
        z_data = {}
        z_data['database_utms'] = test_ds.database_utms
        z_data['positives_per_query'] = positives_per_query
        z_data['predictions'] = predictions
        z_data['distances'] = distances

        torch.save(z_data, log_dir / "z_data.torch")

if __name__ == "__main__":
    args = parser.parse_arguments()
    main(args)
