# LPSR 0.3.0 — API Python et démonstration de super-résolution (×2 / ×4)

Outil hors ligne qui produit, à partir d'une **séquence d'images** d'un même véhicule (idéalement 10), une **image améliorée** de la plaque pour chaque image, agrandie ×2 (par défaut) ou ×4. Le modèle combine l'information de **toutes les images** de la séquence.

La version **0.3.0** ajoute une API de calcul et un package installable en wheel.
Elle conserve le **réseau et les paramètres du modèle officiel publié en v0.2**.
Les fichiers de poids utilisés avec ce package sont copiés et renommés **v0.3**,
mais leur contenu reste identique : il n'y a ni réentraînement ni nouveau modèle.
Le script de démonstration appelle la même API que les applications intégratrices.

## Ce qui change par rapport à la v0.1

La v0.1 (Real‑ESRGAN ×2, modèle générique) **est retirée**. Elle rend les plaques plus nettes en apparence, mais **invente des caractères** : un caractère peu lisible est remplacé par un autre, affiché nettement (par exemple un « B » rendu comme un « 6 »), et une plaque illisible peut devenir un texte net mais faux. Sur les séquences de test, elle dégrade la lisibilité par rapport à un simple agrandissement. Son script reste disponible dans `legacy/` et sous l'étiquette `v0.1`.

La v0.2 a un autre objectif : être **fidèle** aux images de la caméra.

- **Elle combine les 10 images.** Le détail est ainsi récupéré à partir de ce que la caméra a réellement enregistré, plutôt que d'être inventé à partir d'une seule image.
- **Elle a été entraînée sur de vraies paires basse/haute résolution** (20 000 séquences de plaques réelles filmées par des caméras de surveillance Brésil-Mercosur). Elle a ensuite été adaptée aux caméras françaises : chaque image doit pouvoir être reproduite à partir des autres.
- **Quand l'information manque, l'image reste floue** au lieu d'afficher un caractère inventé : l'incertitude reste visible.

## Résultats mesurés

Évaluation sur 71 séquences réelles de test, jamais utilisées pendant le développement (aucune plaque commune avec l'entraînement). La lisibilité est mesurée par un lecteur automatique de texte, utilisé seulement pour l'évaluation : il ne fait pas partie de l'outil.

| Méthode (×2) | Probabilité du bon caractère¹ | Caractères gagnés / perdus vs bicubique | Plaques lues exactement² |
| --- | --- | --- | --- |
| Bicubique ×2 (référence) | 0,074 | — | 19,7 % |
| v0.1 Real‑ESRGAN ×2 | 0,021 | 4,4 % / 31,2 % | 1,4 % |
| **v0.2 ×2** | **0,119** | **17,1 % / 5,0 %** | **21,1 %** |

¹ Moyenne géométrique, par caractère, de la probabilité que le lecteur attribue au bon caractère. ² Vote du lecteur sur les 10 images de la séquence.

- **Gain significatif.** Par rapport au bicubique, la v0.2 rend le bon caractère plus probable : intervalle de confiance à 95 % de la différence de log‑probabilité par caractère [+0,37 ; +0,59]. La v0.1 est significativement moins bonne : [−1,45 ; −1,11].
- **Environ trois fois plus de caractères gagnés que perdus.**
- **Pas davantage d'erreurs « sûres d'elles »** : aucune hausse significative des caractères faux affichés nettement.

**Limite.** Les plaques font en moyenne environ 50 pixels de large et sont compressées (vidéo). Une grande partie de l'information est absente des images : le gain est réel mais progressif.

## Installation

### Rôle de pyproject.toml et des fichiers requirements

**`pyproject.toml` est la source principale de configuration du package.**
Il contient :

- `[build-system]` : les outils qui construisent le wheel ;
- `[project]` : le nom, la version, la version de Python requise et les dépendances ;
- `[project.optional-dependencies]` : les suppléments, dont `demo` pour Pillow ;
- `[tool.setuptools]` : les modules à embarquer et les fichiers à exclure.

`requirements.txt` ne duplique pas cette liste. Sa ligne active est :

```text
.[demo]
```

Le **`.`** désigne le projet dans le répertoire courant : pip utilise son
`pyproject.toml` pour le construire et l'installer avec ses dépendances.
**`[demo]`** ajoute les dépendances du supplément `demo`, ici Pillow.
Depuis la racine du dépôt, ces commandes sont donc équivalentes :

```powershell
python -m pip install -r requirements.txt
python -m pip install ".[demo]"
```

`requirements-cuda.txt` sélectionne en plus les versions CUDA du démonstrateur
puis inclut `requirements.txt`. La chaîne est donc :

```text
requirements-cuda.txt
  -> torch et torchvision en versions +cu130
  -> requirements.txt : .[demo]
     -> pyproject.toml : package, dépendances communes et supplément demo
```

Enfin, `python -m build --wheel` lit lui aussi `pyproject.toml` pour construire
le wheel. Celui-ci contient le code et les métadonnées des dépendances, pas les
bibliothèques torch/OpenCV elles-mêmes. Lors de son installation, pip utilise
ces métadonnées pour installer ou vérifier les dépendances.

### Depuis les sources

Python **3.13 ou supérieur** est requis. Sous Windows, depuis la racine du dépôt :

```powershell
py -3.13 -m venv venv313
venv313\Scripts\python.exe -m pip install --upgrade pip
venv313\Scripts\python.exe -m pip install -r requirements.txt
```

Pour retrouver les versions **CUDA 13.0 du démonstrateur Improved**, utiliser
plutôt ce profil d'installation :

```powershell
venv313\Scripts\python.exe -m pip install -r requirements-cuda.txt
```

Ce fichier sélectionne explicitement `torch==2.13.0+cu130` et
`torchvision==0.28.0+cu130` depuis l'index CUDA PyTorch, puis installe le package.
Ces versions exactes servent à reproduire l'environnement d'intégration ;
les dépendances déclarées dans le wheel restent non figées pour torch et
torchvision. Des paquets CUDA installés ne suffisent pas à établir que le GPU
est utilisable : vérifier `torch.cuda.is_available()` sur la machine d'exécution.

`requirements.txt` installe `.[demo]` : les dépendances sont définies dans
`pyproject.toml`, avec Pillow en supplément pour la planche de démonstration.
Pour un développement avec prise en compte immédiate des changements :

```powershell
venv313\Scripts\python.exe -m pip install -e ".[demo]"
```

Sous Linux, utiliser `python3.13 -m venv venv313` puis `venv313/bin/python` pour
les mêmes commandes. Choisir les distributions torch et torchvision adaptées
au matériel (CPU ou CUDA). Le package n'impose pas de distribution CUDA.

### Construire et installer le wheel

```powershell
venv313\Scripts\python.exe -m pip install build
venv313\Scripts\python.exe -m build --wheel
```

Le fichier généré est :

```text
dist/improved_lpsr-0.3.0-py3-none-any.whl
```

Installation dans l'environnement de l'application appelante :

```powershell
python -m pip install chemin/vers/improved_lpsr-0.3.0-py3-none-any.whl
python -c "from lpsr import LPSREnhancer, __version__; print(__version__)"
```

Le wheel contient uniquement le package de calcul `lpsr`. Les dépendances
natives sont installées séparément par pip ; le nom `py3-none-any` concerne
le code Python de ce package. Le script de démonstration reste dans le dépôt.
Le dossier `legacy/`, les poids, les images et les résultats ne sont pas inclus.

### Versions de dépendances

Les références sont alignées sur l'environnement du démonstrateur Improved :

| Dépendance | Déclaration du package | Référence de validation |
| --- | --- | --- |
| Python | ≥ 3.13 | 3.13.12 |
| torch | ≥ 2.13.0, non figé | 2.13.0+cu130 |
| torchvision | ≥ 0.28.0, non figé | 0.28.0+cu130 |
| NumPy | 2.3.5 | 2.3.5 |
| opencv-python-headless | 5.0.0.93 | 5.0.0.93 |
| Pillow (supplément `demo`) | 12.1.0 | 12.1.0 |

L'API utilise OpenCV **headless**, sans dépendance à une interface graphique.
Éviter d'installer plusieurs variantes d'OpenCV dans le même environnement,
car elles fournissent toutes le module `cv2`.

### Poids du modèle

Les exemples d'intégration utilisent le dossier **`modeles/v0.3`** et les fichiers :

```text
modeles/v0.3/
  weights_lpsr_v0.3_x2.pth
  weights_lpsr_v0.3_x4.pth
  SHA256SUMS
```

Ce sont des **copies renommées des poids officiels v0.2**, sans modification
du contenu ni réentraînement. Le nom `v0.3` les associe à la version du package ;
il ne désigne pas une nouvelle release de poids publiée par XLIM.
Adapter le chemin `modeles/v0.3` à l'emplacement réel des fichiers.

Pour préparer ce dossier à partir des fichiers officiels :

1. **Télécharger les poids** depuis la page *Releases* du dépôt (tag `v0.2`) :
   - `weights_lpsr_v0.2_x2.pth` (×2, 35 Mo)
   - `weights_lpsr_v0.2_x4.pth` (×4, 37 Mo)

2. **Vérifier les fichiers officiels** avec `SHA256SUMS` (fourni avec la release) :

   ```bash
   sha256sum -c SHA256SUMS
   ```

3. Les copier dans le dossier `v0.3`, puis renommer les fichiers :
   - `weights_lpsr_v0.2_x2.pth` → `weights_lpsr_v0.3_x2.pth` ;
   - `weights_lpsr_v0.2_x4.pth` → `weights_lpsr_v0.3_x4.pth`.
   Adapter également les deux noms dans la copie de `SHA256SUMS`, en conservant
   les empreintes inchangées. La vérification peut alors être répétée dans `v0.3`.

Aucune connexion Internet n'est nécessaire ensuite. Une carte graphique est recommandée ; sur processeur (`--device cpu`), compter environ 5 secondes par image.

Les poids restent **hors du wheel**. L'API reçoit leur chemin explicitement et
les charge avec `torch.load(..., weights_only=True)`, sans téléchargement
automatique. Pour la démonstration, utiliser `--weights` avec le chemin du
fichier `v0.3` choisi. Les noms officiels `v0.2` restent également acceptés.

## API de calcul pour intégration

```python
import cv2
from lpsr import LPSREnhancer

enhancer = LPSREnhancer("modeles/v0.3/weights_lpsr_v0.3_x2.pth", device=None)
print(enhancer.scale)  # 2 ou 4, lu dans model_cfg du fichier de poids

crops = [cv2.imread("recadrage_01.png"), cv2.imread("recadrage_02.png")]
results = enhancer.enhance(crops)
# Réutiliser enhancer pour les séquences suivantes sans recharger les poids.
```

- `device=None` choisit CUDA si disponible, sinon CPU. `device="cpu"` force le CPU.
  Un périphérique explicitement demandé n'est pas remplacé silencieusement.
- Le facteur provient de `model_cfg.scale` dans les poids. Son absence conserve
  le défaut ×2 du réseau, comme pour le checkpoint ×2 de la release v0.2.
- `crops` est une liste ou un tuple non vide d'images NumPy **BGR uint8 H×W×3**
  de la **même plaque**, avec les marges décrites ci-dessous.
- Les recadrages peuvent avoir des dimensions différentes. Chacun devient à son
  tour l'image de référence ; les autres sont redimensionnés comme dans la v0.2.
- La méthode renvoie une liste de même longueur et dans le même ordre, contenant
  des images **BGR uint8**, de dimensions `(H × scale, W × scale, 3)` pour chaque
  recadrage d'entrée. Les entrées ne sont pas modifiées.
- La classe conserve le modèle chargé. Le calcul ne lit ni n'écrit de fichiers
  images et ne crée ni fenêtre ni planche comparative.
- Les chemins de poids inexistants, checkpoints invalides et images de format
  incorrect provoquent des exceptions explicites ; aucun chargement non sécurisé
  n'est tenté après un échec de `weights_only=True`.

### Recadrage à réaliser en amont

L'application appelante prépare les recadrages. Pour une boîte de plaque de
largeur `w` et de hauteur `h`, conserver les marges utilisées à l'entraînement :

- **25 % de `w` à gauche et à droite** ;
- **50 % de `h` en haut et en bas** ;
- limiter le rectangle obtenu aux bords de l'image.

Le recadrage fait donc normalement `1,5 × w` par `2 × h`, sauf limitation aux
bords. La fonction `crop_plate` du script de démonstration reste la référence
de ce calcul. L'API ne rajoute pas une seconde marge. Une marge différente peut
dégrader la qualité du modèle.

## Utilisation

```powershell
venv313\Scripts\python.exe enhance_sequence.py --input dossier_sequence --output resultats --weights modeles/v0.3/weights_lpsr_v0.3_x2.pth
venv313\Scripts\python.exe enhance_sequence.py --input dossier_sequence --output resultats --weights modeles/v0.3/weights_lpsr_v0.3_x4.pth
venv313\Scripts\python.exe enhance_sequence.py --input dossier_sequence --output resultats --weights modeles/v0.3/weights_lpsr_v0.3_x2.pth --device cpu
```

Exécuter ces commandes depuis la racine du dépôt. Remplacer `dossier_sequence`
par le dossier d'entrée et adapter le chemin des poids à leur emplacement réel.
Les fichiers sont écrits dans `resultats`, avec `planche.png` pour les comparer.
La démo produit des fichiers ; elle n'ouvre pas de fenêtre graphique.

Avec `--weights`, le facteur est automatiquement lu dans le checkpoint.
Si `--scale` est également fourni, il doit correspondre au modèle chargé.
Pour conserver la compatibilité avec la démonstration d'origine, sans `--weights`,
`--scale` choisit encore les fichiers officiels nommés v0.2 placés à côté du
script (×2 par défaut). Avec les copies nommées v0.3, fournir `--weights` comme
dans les exemples ci-dessus. La lecture des fichiers, le recadrage, les exports et la planche
comparative sont propres à cette démonstration ; ils ne font pas partie de l'API.

Le dossier d'entrée contient :

- **soit** les images complètes et un fichier `detections.json` : `[{"frame": "000000.png", "license_plate_coordinates": [x1, y1, x2, y2]}, ...]`. Vérifier que les coordonnées correspondent bien aux images : sur certaines caméras, les images avaient été rognées en haut sans que les coordonnées soient corrigées ;
- **soit** uniquement des recadrages de la plaque **avec une marge** (plaque ≈ 2/3 centraux en largeur et 1/2 central en hauteur).

Sortie :

| Fichier | Contenu |
| --- | --- |
| `plaque_XX_originale.png` | recadrage original de l'image XX |
| `plaque_XX_bicubique_xS.png` | agrandissement classique (référence neutre) |
| `plaque_XX_amelioree_xS.png` | amélioration multi-images |
| `planche.png` | planche comparative avec l'avertissement (désactivable : `--no_sheet`) |

## Contenu du dépôt

- `enhance_sequence.py` – outil en ligne de commande
- `lpsr/enhancer.py` – API `LPSREnhancer` et chargement unique des poids
- `lpsr/inference.py` – calcul sur des tableaux d'images, sans présentation
- `lpsr/model.py` – définition du réseau
- `lpsr/_version.py` – version du package, également utilisée pour le wheel
- `pyproject.toml` – métadonnées, dépendances et construction du package
- `requirements.txt` – installation du package et des dépendances de démonstration
- `requirements-cuda.txt` – profil des versions CUDA 13.0 du démonstrateur
- `tests/` – tests synthétiques de l'API et du programme de démonstration
- `legacy/srmodel_enhance_v0.1.py` – ancienne version (retirée)

## Tests

Depuis la racine du dépôt, après installation de `.[demo]` :

```powershell
venv313\Scripts\python.exe -m unittest discover -s tests -v
```

Les tests créent des images synthétiques et de petits checkpoints temporaires.
Ils vérifient les facteurs ×2/×4, les formats et l'ordre des sorties, le chargement
unique et sécurisé, les erreurs d'entrée, les marges et l'appel de l'API par la
démonstration. Aucun poids de production n'est nécessaire pour ces tests.

Pour comparer numériquement des sorties CPU et GPU, les convertir en entiers
signés avant soustraction afin d'éviter le débordement uint8. La tolérance de
comparaison est un écart absolu maximal de **1 sur l'échelle 0–255**, sur tous
les pixels et canaux. Documenter séparément le périphérique et les versions
effectivement testés : un contrôle CPU ne constitue pas une validation GPU.

## Détails techniques

- **Réseau :**
  - encodeur par image (4 blocs d'attention de canal) ;
  - alignement déformable sur l'image de référence ;
  - fusion temporelle pondérée par la similarité avec la référence ;
  - reconstruction (5 groupes × 8 blocs) puis agrandissement ×2 ou ×4.

  96 canaux, 8,8 M paramètres. Chaque image de la séquence sert à tour de rôle de référence.
- **Entraînement :**
  - paires réelles basse / haute résolution du jeu d'entraînement de la compétition ICPR 2026 *Low‑Resolution License Plate Recognition* (plaques brésiliennes et Mercosur), avec une perte de reconstruction tolérante aux petits décalages et une perte de lisibilité (lecteur figé) ;
  - adaptation aux caméras françaises par auto‑supervision **sans étiquette** : l'image restaurée, recalée et réduite, doit reproduire une image retirée de la séquence.
- **Pourquoi l'adaptation française est indispensable** : entraîné sur les seules données ICPR, le modèle redessine les plaques françaises « à la brésilienne » et invente des caractères.
- **Évaluation** : 474 séquences françaises réelles, réparties par plaque (aucune plaque commune entre entraînement, validation et test). Les réglages ont été choisis sur la validation ; le test n'a servi qu'à la mesure finale.

## Licence

Code : MIT. Le modèle a été entraîné en partie sur les données ICPR 2026 LRLPR : vérifier leurs conditions d'utilisation avant tout usage opérationnel.
