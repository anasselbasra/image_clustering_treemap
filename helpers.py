############################################################################
# Fichier: computer_vision/vision_models.py
# Auteur: Anass El Basraoui
# Date: 2025-10-08
# Description: Fonctions et modèles de vision par ordinateur avec gestion des téléchargements d'images.
############################################################################


from urllib.parse import urlparse # Pour extraire le nom de fichier à partir d’une URL.
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests, time, os
from pathlib import Path  # Gère proprement les chemins de fichiers (OS-independent)
import pandas as pd
import numpy as np 
import hdbscan
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import torch
from PIL import Image
from io import BytesIO
import logging
from transformers import (
    AutoImageProcessor,  # Gère automatiquement les pré-traitements (resize, normalisation...) selon le modèle
    AutoModel,           # Permet de charger les modèles de vision (ex : DINOv2, SigLIP)
    AutoProcessor,
    CLIPProcessor,       # Processor spécifique pour les modèles CLIP (image + texte)
    CLIPModel           # Modèle CLIP complet, utilisé ici uniquement pour la partie image
)
logging.getLogger("transformer").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.WARNING)


device = "cuda" if torch.cuda.is_available() else "cpu"

def show_image(path):
    from  matplotlib.pyplot import  imread, imshow, axis, show
    # Chemin vers votre image
    chemin_image = path
    # Lire l'image
    img = imread(chemin_image)

    # Afficher l'image
    imshow(img)
    axis('off')  # Masquer les axes pour une vue propre
    show()





def download_with_retry(url, path, folder_name, retries=2, delay=1, timeout=8):
    """
    Download an image from a given URL with multiple retry attempts.

    This function attempts to download an image from a remote URL and save it
    into a local directory. If the download fails (due to timeout, network issues,
    or corrupted content), it automatically retries a given number of times
    before returning an error message.

    Parameters
    ----------
    url : str
        The direct URL of the image to download.
    path : str or Path
        The base directory where the subfolder will be created.
    folder_name : str
        The name of the subfolder inside `path` where the image will be saved.
    retries : int, optional (default=2)
        The number of retry attempts in case of failure.
    delay : int or float, optional (default=1)
        The delay (in seconds) between retry attempts. The actual waiting time
        increases linearly with each attempt (delay * attempt).
    timeout : int or float, optional (default=8)
        Maximum time (in seconds) to wait for a server response before aborting
        the request.

    Returns
    -------
    tuple
        A 3-element tuple in the form:
        `(url, local_path or None, error_message or None)`

        - `url`: The original URL of the image.  
        - `local_path`: Full path to the saved image file if successful, otherwise `None`.  
        - `error_message`: `None` if success, otherwise a short string describing the error.

    Notes
    -----
    - The image is always converted to RGB format before saving.
    - If the URL does not contain a valid file name, a fallback name is automatically generated.
    - The save directory is created if it does not exist.

    Examples
    --------
    >>> download_with_retry(
    ...     url="https://pbs.twimg.com/media/example.jpg",
    ...     path="/data/images",
    ...     folder_name="dataset1"
    ... )
    ('https://pbs.twimg.com/media/example.jpg',
     '/data/images/dataset1/example.jpg',
     None)
    """
    # --- Prepare target folder ---
    save_dir = os.path.join(path, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Derive a valid local filename ---
    filename = os.path.basename(urlparse(url).path) or f"img_{int(time.time() * 1000)}.jpg"
    local_path = os.path.join(save_dir, filename)

    # --- Attempt multiple retries if download fails ---
    for attempt in range(1, retries + 1):
        try:
            # Download and validate HTTP response
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()

            # Open and save image locally
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(local_path)
            return (url, local_path, None)

        except Exception as e:
            err = f"{type(e).__name__}: {e}"

            # Retry with exponential backoff
            if attempt < retries:
                time.sleep(delay * attempt)
            else:
                # Final failure: return the error
                return (url, None, err)


# Exécution parallèle et collecte des résultats
def parallel_download(urls, path, folder_name, return_failed_csv=False, csv_name="failed_urls.csv", timeout=10):
    """
    Download multiple images in parallel with automatic retry and failure tracking.

    This function handles large-scale image downloads efficiently using a
    multi-threaded approach (`ThreadPoolExecutor`). Each image is downloaded
    via the helper function `download_with_retry`, which automatically retries
    failed requests. Failed downloads can optionally be logged in a CSV file.

    Parameters
    ----------
    urls : list of str
        List of image URLs to download.
    path : str or Path
        Root directory where images will be saved. A subfolder (`folder_name`)
        will be created inside this directory.
    folder_name : str
        Name of the subfolder (typically the dataset name or topic).
    return_failed_csv : bool, optional (default=False)
        If True, creates a CSV file logging all failed downloads.
    csv_name : str, optional (default="failed_urls.csv")
        Name of the CSV file that will store failed downloads (created in `path`).
    timeout : int or float, optional (default=10)
        Timeout (in seconds) for each image request.

    Returns
    -------
    results : list of tuple
        List of `(url, local_path, error)` tuples:
        - `url` (str): the original image URL  
        - `local_path` (str or None): the local file path if downloaded successfully  
        - `error` (str or None): error message if download failed  

    failed : list of tuple
        List of `(url, error)` tuples for all images that failed to download.

    Notes
    -----
    - This function automatically adapts the number of threads to the number
      of available CPU cores (`os.cpu_count()`).
    - If `return_failed_csv=True`, a CSV file named `failed_urls.csv` (by default)
      is created in the target folder.
    - It is recommended to run this function **only once** per dataset, as the
      downloaded images can be reused later.

    Examples
    --------
    >>> urls = ["https://pbs.twimg.com/media/example1.jpg",
    ...         "https://pbs.twimg.com/media/example2.jpg"]
    >>> results, failed = parallel_download(
    ...     urls,
    ...     path="/data/images",
    ...     folder_name="10_septembre",
    ...     return_failed_csv=True,
    ...     timeout=3
    ... )
    Total URLs: 2
    Succès: 2
    Échecs: 0
    """
    results = []
    failed = []
    max_workers = os.cpu_count() or 8

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(download_with_retry, url, path, folder_name, timeout=timeout): url
            for url in urls
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Téléchargement"):
            url = futures[fut]
            try:
                url, local_path, error = fut.result()
                results.append((url, local_path, error))
                if error:
                    failed.append((url, error))
            except Exception as e:
                failed.append((url, f"FutureError: {e}"))

    # --- Log failed URLs if requested ---
    if return_failed_csv and failed:
        failed_csv = os.path.join(path, csv_name)
        with open(failed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["url", "error"])
            writer.writerows(failed)
        print(f"Fichier des échecs enregistré: {failed_csv}")

    # --- Summary statistics ---
    print(f"Total URLs: {len(urls)}")
    print(f"Succès: {len([r for r in results if r[1]])}")
    print(f"Échecs: {len(failed)}" + (f" (voir {failed_csv})" if return_failed_csv else ""))

    return results, failed



##################################################################################################
## Application d'embedding avec un modèle de vision
##################################################################################################

################################################ DINO ###############################################
def encode_with_dino(image_dir, processor, model, device="cuda", batch_size=16):
    """
    Encode toutes les images d'un dossier avec un modèle visuel (ex : DINOv2).
    Retourne un DataFrame (filename, embedding).
    """
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = [p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts]

    embeddings, names = [], []
    max_workers = os.cpu_count() or 8
    for i in tqdm(range(0, len(paths), batch_size), desc="Encoding batches"):
        batch_paths = paths[i:i+batch_size]

        # Chargement parallèle des images sert à accélérer la lecture des images depuis le disque vers la RAM.
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            images = list(ex.map(lambda p: Image.open(p).convert("RGB"), batch_paths)) 

        # Ici commence le role de la device
        # Prétraitement des images + conversion en float16
        inputs = processor(images=images, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"]  # garantit compatibilité FP16

        # Encodage
        with torch.no_grad():
            out = model(**inputs)
            if hasattr(out, "pooler_output"):
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state.mean(dim=1)
            else:
                raise ValueError("Modèle non compatible : pas de pooler_output ni last_hidden_state")

            feats = torch.nn.functional.normalize(feats, dim=-1)

        embeddings.extend(feats.cpu().float().numpy())
        names.extend([p.name for p in batch_paths])

    return pd.DataFrame({"filename": names, "embedding": embeddings})

def upload_dino(model_name="facebook/dinov2-large", device="cuda"):
    """
    Charge automatiquement un modèle visuel (DINOv2, v3, etc.)
    et renvoie (processor, model) sur le bon device.
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()

    print(f"Modèle: {model_name} chargé sur {device}")

    return processor, model

########################################################################################################
################################################ CLIP ###############################################

## Econdage d'une image avec un modèle de vision
def encode_with_clip(image_dir=None, texts=None, processor=None, model=None, device=device, batch_size=16):
    """
    Encode images and/or texts using a CLIP or OpenCLIP model.

    This function generates vector embeddings for a set of images, a list of texts,
    or both, using a pre-trained CLIP model. The embeddings are normalized so that
    cosine similarity can be used directly to measure semantic proximity.

    Parameters
    ----------
    image_dir : str, list, or Path, optional
        Path to a directory containing images, a single image file, or a list of image paths.
        Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`.
        If None, only text encoding will be performed.
    texts : str or list of str, optional
        Text(s) to encode. Can be a single string or a list of sentences.
        If None, only image encoding will be performed.
    processor : transformers.CLIPProcessor
        The processor associated with the CLIP model, handling preprocessing steps
        like resizing, normalization, and tokenization.
    model : transformers.CLIPModel
        The pre-trained CLIP model used to generate embeddings.
    device : str, optional
        Computation device, either `"cuda"` or `"cpu"`. Default is the global `device` variable.
    batch_size : int, optional (default=16)
        Number of images processed in each batch during encoding.

    Returns
    -------
    dict
        Dictionary with two DataFrames:

        - `"image_embeddings"`: DataFrame with columns:
            - `filename` : str, name of each processed image file  
            - `embedding` : np.ndarray, the 1×D embedding vector for each image  

        - `"text_embeddings"`: DataFrame with columns:
            - `text` : str, the input text string  
            - `embedding` : np.ndarray, the 1×D embedding vector for each text  

    Notes
    -----
    - All embeddings are L2-normalized (`torch.nn.functional.normalize`), allowing
      cosine similarity to be computed via simple dot product.
    - The function handles both directory and list inputs for image paths.
    - Use a GPU (`device='cuda'`) for much faster computation.
    - Images are automatically converted to RGB before encoding.

    Examples
    --------
    >>> processor, model = upload_clip("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    >>> result = encode_with_clip(image_dir="data/images_10_septembre", processor=processor, model=model)
    >>> result["image_embeddings"].head()

    >>> result = encode_with_clip(texts=["protest", "Macron speech"], processor=processor, model=model)
    >>> result["text_embeddings"].head()
    """
    # Validate input
    if image_dir is None and texts is None:
        raise ValueError("You must provide either 'image_dir' or 'texts' as input.")

    # Initialize result structure
    results = {
        "image_embeddings": pd.DataFrame(),
        "text_embeddings": pd.DataFrame()
    }
    max_workers = os.cpu_count() or 8
    # === IMAGE ENCODING ======================================================
    if image_dir:
        print("Preparing image paths...")
        paths = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        # Handle list input or single directory/file
        if isinstance(image_dir, list):
            paths = [Path(p) for p in image_dir if Path(p).suffix.lower() in valid_exts]
        else:
            p = Path(image_dir)
            if p.is_dir():
                paths = [img_p for img_p in p.iterdir() if img_p.suffix.lower() in valid_exts]
            elif p.is_file() and p.suffix.lower() in valid_exts:
                paths = [p]

        if not paths:
            print("Warning: No valid images found to process.")
        else:
            img_embeddings, img_names = [], []
            for i in tqdm(range(0, len(paths), batch_size), desc="Encoding image batches"):
                batch_paths = paths[i:i + batch_size]
                # Load images concurrently
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    images = list(ex.map(lambda p: Image.open(p).convert("RGB"), batch_paths))

                # Preprocess batch
                inputs = processor(images=images, return_tensors="pt").to(device)

                # Encode images
                with torch.no_grad():
                    feats = model.get_image_features(**inputs)
                    feats = torch.nn.functional.normalize(feats, dim=-1)

                img_embeddings.extend(feats.cpu().float().numpy())
                img_names.extend([p.name for p in batch_paths])

            results["image_embeddings"] = pd.DataFrame({
                "filename": img_names,
                "embedding": img_embeddings
            })

    # === TEXT ENCODING =======================================================
    if texts:
        print("Encoding text...")
        if isinstance(texts, str):
            texts = [texts]

        inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = torch.nn.functional.normalize(feats, dim=-1)

        text_embeddings = feats.cpu().float().numpy()
        results["text_embeddings"] = pd.DataFrame({
            "text": texts,
            "embedding": list(text_embeddings)
        })

    return results





def upload_clip(model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", device="cuda"):
    """
    Load a pre-trained CLIP model and its corresponding processor on the specified device.

    This function automatically loads both the processor and model components of a CLIP
    or OpenCLIP architecture from the Hugging Face Hub, and prepares them for inference.
    The model is moved to the selected device (GPU or CPU) and set to evaluation mode.

    Parameters
    ----------
    model_name : str, optional
        The name or path of the CLIP model checkpoint to load.
        Defaults to `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"`, one of the best-performing
        open CLIP variants with 1024-dimensional embeddings.

        Common examples:
        - `"laion/CLIP-ViT-H-14-laion2B-s32B-b79K"` → 1024-D (best performance)
        - `"openai/clip-vit-large-patch14-336"` → 768-D
        - `"openai/clip-vit-large-patch14"` → 768-D

    device : str, optional
        The target device for inference: `"cuda"` or `"cpu"`. Default is `"cuda"`.
        Automatically uses GPU if available for faster embedding generation.

    Returns
    -------
    tuple
        `(processor, model)` where:
        - `processor` : `transformers.CLIPProcessor`  
          Handles image preprocessing and text tokenization.  
        - `model` : `transformers.CLIPModel`  
          The pre-trained model ready for inference on the selected device.

    Notes
    -----
    - The model is loaded in evaluation mode (`.eval()`).
    - The precision is set to `torch.bfloat16`, which provides a balance between
      numerical stability and GPU performance.
    - The function supports both OpenAI and LAION OpenCLIP checkpoints.

    Examples
    --------
    >>> processor, model = upload_clip()
    Modèle laion/CLIP-ViT-H-14-laion2B-s32B-b79K chargé sur cuda

    >>> processor, model = upload_clip("openai/clip-vit-large-patch14-336", device="cpu")
    Modèle openai/clip-vit-large-patch14-336 chargé sur cpu
    """
    # Load processor and model
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    model = CLIPModel.from_pretrained(model_name, dtype=torch.bfloat16).to(device).eval()

    print(f" Modèle {model_name} chargé sur {device}")
    return processor, model

### "openai/clip-vit-large-patch14" <- 768 <- 3
### "laion/CLIP-ViT-H-14-laion2B-s32B-b79K" <- 1024 <- 1 (le meilleur) 
### "openai/clip-vit-large-patch14-336"<- 768 <-  2

########################################################################################################
########################################################################################################

######   Outil 1: SEARCH ENGINE

########################################################################################################
########################################################################################################

def search_engine(image_dir, query, df, model, processor, k=10, device="cuda"):
    """
    Search for the most semantically similar images to a given query.

    This function retrieves the top-k most similar images from a precomputed
    index of image embeddings. The query can be a **text string**, a **path to an image**, 
    or an **embedding vector**. The similarity is computed using the cosine similarity
    between the query embedding and the image embeddings in the dataset.

    Parameters
    ----------
    image_dir : str or Path
        Path to the folder containing all downloaded images.
    query : str or np.ndarray
        The search query. It can be:
          - a text string (e.g. `"Trump"`)
          - a path to a local image file (e.g. `"images/trump.png"`)
          - a NumPy array representing a precomputed embedding
    df : pandas.DataFrame
        DataFrame containing at least two columns:
        - `"filename"` : name of each image file  
        - `"embedding"` : embedding vector (np.ndarray)
    model : transformers.CLIPModel
        The CLIP model used for encoding text or image queries.
    processor : transformers.CLIPProcessor
        The corresponding processor used for preprocessing (tokenization, resizing, normalization).
    k : int, optional (default=10)
        Number of top similar images to retrieve.
    device : str, optional (default="cuda")
        Computation device (`"cuda"` or `"cpu"`).

    Returns
    -------
    None
        The function displays the top-k similar images directly using Matplotlib.

    Notes
    -----
    - Cosine similarity is computed as the **dot product** between normalized embeddings.
    - The function automatically detects whether the query is a text, an image path,
      or a precomputed embedding.
    - Works seamlessly with embeddings generated via `encode_with_clip()`.
    - This function does not return values; it displays the retrieved images.

    Examples
    --------
    >>> # Text-based search
    >>> search_engine(
    ...     image_dir="data/img_data/10_septembre",
    ...     query="Jean-Luc Mélenchon",
    ...     df=df,
    ...     model=model,
    ...     processor=processor,
    ...     k=5
    ... )

    >>> # Image-based search
    >>> img_path = "data/img_data/10_septembre/G0_aQP2WoAA9nbX.jpg"
    >>> search_engine(
    ...     image_dir="data/img_data/10_septembre",
    ...     query=img_path,
    ...     df=df,
    ...     model=model,
    ...     processor=processor,
    ...     k=5
    ... )

    >>> # Embedding-based search
    >>> query_vec = np.random.randn(1024)
    >>> search_engine(
    ...     image_dir="data/img_data/10_septembre",
    ...     query=query_vec,
    ...     df=df,
    ...     model=model,
    ...     processor=processor,
    ...     k=5
    ... )
    """
    # --- 1. Get query embedding ---
    query_embedding = None

    # Case 1: Query is already an embedding
    if isinstance(query, np.ndarray):
        query_embedding = query

    # Case 2: Query is text or image path
    elif isinstance(query, str):
        # Detect if it's an image file
        is_image_path = os.path.exists(query) and query.split('.')[-1].lower() in ["jpg", "jpeg", "png", "webp"]

        if is_image_path:
            print("Searching by image...")
            results = encode_with_clip(processor=processor, model=model, image_dir=query)
            query_embedding = results["image_embeddings"]["embedding"].iloc[0]
        else:
            print("Searching by text...")
            results = encode_with_clip(processor=processor, model=model, texts=query)
            query_embedding = results["text_embeddings"]["embedding"].iloc[0]

    if query_embedding is None:
        raise ValueError("Unrecognized query format. Must be text, image path, or embedding vector.")

    # --- 2. Compute cosine similarities ---
    image_matrix = np.vstack(df["embedding"].values)
    similarities = image_matrix @ query_embedding.T  # since embeddings are normalized

    # --- 3. Get top-k results ---
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_files = df["filename"].iloc[top_k_indices].tolist()
    top_scores = similarities[top_k_indices]

    # --- 4. Display results ---
    cols = 5
    rows = (k + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (fname, score) in enumerate(zip(top_files, top_scores)):
        img_path = Path(image_dir) / fname
        img = Image.open(img_path).convert("RGB")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{fname}\nSimilarity: {score:.3f}", fontsize=12)

    plt.tight_layout()
    plt.show()



















########################################################################################################
## Outil 2: CLustering
########################################################################################################

## Test

def test_performance(X, df,start=2, end=200,  step=5):
    _ = df.copy()
    for i in range(start, end, step):
        print(f"{i=}")   
        clusterer = hdbscan.HDBSCAN(min_cluster_size=i, min_samples=5, metric="euclidean")
        labels = clusterer.fit_predict(X)
        _["cluster"] = labels.astype(str)

        n_clusters = len(np.unique(labels[labels >= 0]))
        noise_ratio = len(_[_["cluster"] == "-1"]) / len(_) * 100
        persistence = clusterer.cluster_persistence_.mean() if n_clusters > 0 else np.nan

        # Calcul du silhouette en excluant le bruit
        silhouette = np.nan
        mask = labels != -1
        if n_clusters > 1 and mask.sum() > 1:
            try:
                silhouette = silhouette_score(X[mask], labels[mask])
            except Exception:
                pass

        print(f"Clusters: {n_clusters}, Bruit: {noise_ratio:.2f}%, "
              f"Persistance: {persistence:.3f}, Silhouette: {silhouette:.3f}")