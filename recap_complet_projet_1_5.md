# Projet 1.5 : Compression de modèle point cloud (Teeth3DS+)
## Récapitulatif complet, du début à aujourd'hui (31 octobre 2026)

Ce document reprend **tout** ce qui a été fait et appris, dans l'ordre : données, modèles, distillation, compression (pruning, INT8), mesures CPU, présentation, courbes d'apprentissage, distillation des couches intermédiaires, vocabulaire, problèmes rencontrés et ce qu'on peut affirmer. Les parties 0 à 6 sont le guide initial (données, modèles, baselines, distillation sur les logits) ; les parties 7 et suivantes couvrent la suite. Les scripts complets sont sur ton disque ; ici, des extraits expliqués et les résultats.

**Règle appliquée partout** : on distingue ce qui est **mesuré** de ce qui est **hypothèse**, et on choisit sur la **validation**, jamais sur le test.

---

## 0. L'idée générale en 5 lignes

- **Le but** : un petit modèle d'IA qui reconnaît les dents sur un scan 3D de bouche, assez léger pour tourner sur un PC **sans carte graphique**.
- **Le plan** : (1) préparer les données, (2) entraîner un gros modèle précis mais lent (le *Teacher*, DGCNN) et un petit modèle rapide (le *Student*, PointNet), (3) essayer de faire apprendre le petit au gros (*distillation*), (4) réduire encore le petit (pruning, quantification INT8), (5) mesurer vitesse et précision sur CPU.
- **La tâche d'IA** : *segmentation sémantique de points*. Le modèle reçoit un nuage de points et doit dire, **pour chaque point**, s'il appartient à la gencive ou à quelle dent.

### Où tu en es (31 octobre 2026)

| Étape du plan | État |
|---|---|
| S2 : données + modèles initiaux (Teacher et Student) | **Fait** |
| S3 : distillation sur les logits | **Fait, résultat négatif** (pas de gain) |
| S3 : distillation des couches intermédiaires | **Lancée sur 1 graine** : signal positif non confirmé (partie 11) |
| S4 : pruning non structuré | **Fait** : aucun gain de taille ni de vitesse sur CPU |
| S4 : pruning structuré | **Fait** : vrais gains de latence, taille et mémoire |
| S4 : quantification INT8 | **Fait** (dynamique seulement) |
| S5 : benchmark CPU (latence, taille, RAM), front de Pareto | **Mesures faites** ; rapport final **à écrire** |
| Présentation Beamer semaines 3-4 + speech | **Faits** (17 diapositives) |

### Mini-glossaire

| Mot | Sens simple |
|---|---|
| **Nuage de points** | Une liste de points (x, y, z) : la forme 3D sans les triangles. |
| **Label / étiquette** | La bonne réponse pour un point (gencive ou numéro de dent). |
| **Logits** | Les scores bruts que le modèle sort pour chaque classe, avant de devenir des probabilités. |
| **Époque (epoch)** | Un passage complet du modèle sur toutes les données d'entraînement. |
| **Batch** | Un petit paquet de scans traités ensemble. |
| **Graine (seed)** | Un nombre qui fixe le hasard. Changer de graine = un autre entraînement, ce qui mesure la variabilité. |
| **Validation** | Des scans mis de côté pour choisir/régler. **Test** : d'autres scans, regardés une fois à la fin. |
| **OA** | Overall Accuracy : % de points correctement classés. |
| **mIoU** | Moyenne, sur les classes, du recouvrement entre prédiction et réalité. C'est la métrique la plus honnête ici. |
| **Surapprentissage** | Le modèle est meilleur sur ce qu'il a déjà vu que sur des données nouvelles. |
| **Teacher / Student** | Le gros modèle précis / le petit modèle léger. |
| **Distillation (KD)** | Entraîner le Student avec les réponses du Teacher en plus des vraies étiquettes. |

---

## 1. Les données : Teeth3DS+

### 1.1 Ce que c'est

Un benchmark public de scans dentaires 3D réalisés par scanner optique intra-oral. D'après la page officielle : **1800 scans de 900 patients** (mâchoire du haut et du bas scannées séparément) et environ 24 000 dents annotées. Source : https://crns-smartvision.github.io/teeth3ds (téléchargement via OSF).

### 1.2 Un échantillon = deux fichiers

Un échantillon est **une arcade d'un patient**. Il a deux fichiers :

| Fichier | Contenu | Rôle |
|---|---|---|
| `ID_lower.obj` / `ID_upper.obj` | Maillage 3D : sommets (x, y, z) + triangles | Les **entrées** du modèle |
| `ID_lower.json` / `ID_upper.json` | `labels` (un par sommet), `instances`, `jaw`, `id_patient` | Les **réponses attendues** |

Exemple minuscule (6 sommets) pour comprendre l'alignement :

| n° sommet | x | y | z | label |
|---|---|---|---|---|
| 0 | 2.1 | -14.6 | -82.4 | 0 (gencive) |
| 1 | 2.5 | -14.1 | -81.9 | 0 (gencive) |
| 2 | 3.0 | -13.8 | -82.0 | 44 (dent 44) |
| 3 | 4.2 | -12.0 | -80.5 | 44 |
| 4 | 4.8 | -11.5 | -80.1 | 45 |
| 5 | 5.1 | -11.0 | -79.7 | 0 |

Le tableau des coordonnées est de forme `[N, 3]`, celui des labels de forme `[N]`, **alignés ligne par ligne** : le sommet *i* a le label *i*. Dans les vrais scans, N vaut des dizaines de milliers (84 000 à 186 000 sur les 3 scans que tu as inspectés).

### 1.3 Les labels : numérotation FDI et mes 17 classes

Le label `0` est la gencive. Les autres valeurs suivent la numérotation dentaire internationale **FDI** (11 à 18, 21 à 28 pour le haut ; 31 à 38, 41 à 48 pour le bas). Ces numéros ne sont pas contigus, alors j'ai ramené tout à **17 classes** (gencive + 16 positions), **partagées entre les deux arcades** :

| FDI | Classe |
|---|---|
| 0 | 0 (gencive) |
| 11–18 ou 31–38 | 1 à 8 |
| 21–28 ou 41–48 | 9 à 16 |

```python
N_PTS, NUM_CLASSES = 4096, 17

def fdi_to_class(l):
    # 0 = gencive ; FDI 11-18 / 31-38 -> classes 1..8 ; FDI 21-28 / 41-48 -> classes 9..16
    if l == 0: return 0
    q, p = divmod(l, 10)
    if q in (1, 3) and 1 <= p <= 8: return p
    if q in (2, 4) and 1 <= p <= 8: return 8 + p
    return 0                                   # label inattendu -> gencive
```

**C'est un choix de conception** (pas une obligation du dataset). Conséquence : le modèle ne distingue pas haut et bas, il prédit seulement « quelle position dans l'arcade ».

### 1.4 Comment les données ont été récupérées (et ce qui a coincé)

1. Téléchargement de **7 archives** `PART1` à `PART7` depuis OSF (≈ 6,4 Go). La page officielle en annonçait 6 : j'ai donc fait vérifier les doublons.
2. **Chaque archive contenait une autre archive** (`data_part_N.zip`) : d'où 0 `.obj` après la première extraction.
3. **PART3 était tronquée** (`End-of-central-directory signature not found`) : re-téléchargée.
4. **PART5** : ma boucle `for` a coupé un chemin contenant un espace (dossier `... (1)`), ce qui a produit un faux « CASSE ». L'archive était saine.
5. **La part 7 n'a pas de `.json`** (100 scans). Ce sont très probablement les scans de test de 3DTeethLand, sans annotations : inutilisables pour entraîner ou évaluer.

Commandes utilisées pour l'extraction (la partie qui a fonctionné) :

```bash
mkdir -p teeth3ds_files
for z in $(find teeth3ds_data -name '*.zip' | sort); do
  n=$(basename "$z" .zip)
  if unzip -tq "$z" >/dev/null 2>&1; then unzip -q -o "$z" -d "teeth3ds_files/$n"; else echo "CASSE $z"; fi
done
# PART5 extraite à part (le chemin contient un espace, donc entre guillemets) :
Z="teeth3ds_data/PART5_7az58-osfstorage-archive (1)/data_part_5.zip"
unzip -tq "$Z" && unzip -q -o "$Z" -d teeth3ds_files/data_part_5
```

Résultat : **1800 paires `.obj` + `.json`** exploitables (le script retrouve les fichiers par leur nom, quelle que soit l'arborescence).

### 1.5 Les découpages entraînement / test, et un piège

Le dataset est fourni avec des listes officielles. J'ai vérifié sur tes listes :

- Split officiel Teeth3DS : 1200 scans en train, 600 en test, aucun scan en commun.
- **Mais il est fait par arcade, pas par patient** : **382 patients sur 900 ont une arcade en train et l'autre en test**. Les deux arcades d'un même patient se ressemblent beaucoup : les scores seraient gonflés (**fuite de données**).
- Les listes 3DTeethLand (340 scans) ne couvrent qu'un sous-ensemble : inutilisables ici.

**Solution retenue** : un découpage **par patient** (70 % / 15 % / 15 %, graine fixe), soit **1260 scans en entraînement, 270 en validation, 270 en test**. Tous les résultats ci-dessous utilisent ce même découpage.

```python
def patient_split(ids, seed=0, ratios=(0.7, 0.15, 0.15)):
    pats = sorted({i.rsplit("_", 1)[0] for i in ids}); random.Random(seed).shuffle(pats)
    a, b = int(ratios[0] * len(pats)), int((ratios[0] + ratios[1]) * len(pats))
    S = [set(pats[:a]), set(pats[a:b]), set(pats[b:])]
    return [[i for i in ids if i.rsplit("_", 1)[0] in s] for s in S]
```

En clair : on mélange les **patients** (pas les scans), puis on met chaque patient avec ses deux arcades dans un seul des trois groupes.

---

## 2. Préparer les données : du scan au nuage de points de 4096 points

### 2.1 Lire un scan (et vérifier l'alignement)

```python
def index_files(root):
    objs = {p.stem: p for p in Path(root).rglob("*.obj")}
    jsons = {p.stem: p for p in Path(root).rglob("*.json")}
    ids = sorted(set(objs) & set(jsons))       # on ne garde que les paires complètes
    return {i: (objs[i], jsons[i]) for i in ids}

def load_scan(obj_path, json_path):
    mesh = trimesh.load(str(obj_path), process=False)   # process=False : garde l'ordre des sommets
    v = np.asarray(mesh.vertices, dtype=np.float32)
    lab = np.asarray(json.load(open(json_path))["labels"])
    assert len(v) == len(lab), f"{Path(obj_path).name}: {len(v)} sommets vs {len(lab)} labels"
    return v, np.array([fdi_to_class(int(x)) for x in lab], dtype=np.int64)
```

- `index_files` associe chaque `.obj` à son `.json` par leur nom.
- `load_scan` lit les sommets (`[N, 3]`) et les labels (`[N]`), et **vérifie qu'il y en a autant**. `process=False` est important : sinon trimesh pourrait fusionner ou réordonner des sommets, et l'alignement serait faux.

Cette vérification a été exécutée sur **les 1800 scans** quand tu as construit le cache : aucune erreur. Ça prouve que *les longueurs* correspondent. Ça ne prouve pas que chaque label est sémantiquement juste (je n'ai rien visualisé).

### 2.2 Ramener chaque scan à 4096 points

Les scans ont des tailles très différentes, et un modèle veut des entrées de **même taille**. On tire donc 4096 sommets au hasard, avec leurs labels, puis on centre et on normalise.

```python
def sample_normalize(v, y, n=N_PTS, rng=np.random):
    idx = rng.choice(len(v), n, replace=len(v) < n)      # mêmes indices pour points ET labels
    p = v[idx]; p = p - p.mean(0); p = p / np.linalg.norm(p, axis=1).max()
    return p.astype(np.float32), y[idx]
```

- Les **mêmes indices** servent aux points et aux labels : l'alignement est conservé.
- `p - p.mean(0)` centre le nuage, la division le ramène dans une sphère de rayon 1.
- Conséquence à connaître : on garde seulement **2 à 5 %** des points d'un scan (4096 sur 84 000 à 186 000). Les détails fins entre deux dents voisines sont en partie perdus.

Résultat par échantillon : `points [4096, 3]` et `labels [4096]`.

### 2.3 Le cache (pour aller vite)

Relire un `.obj` de 180 000 sommets à chaque époque est lent (environ 3 min par époque). `fast.py --mode cache` tire donc **une seule fois** les 4096 points de chacun des 1800 scans et sauvegarde le tout dans `cache_4096.npz` (≈ 100 Mo). Il lui faut ≈ 3 à 4 min, ensuite une époque dure quelques secondes.

```bash
python fast.py --mode cache
```

Le compromis : chaque scan a un échantillon **fixe**. Note : tu as régénéré le cache une fois (nouveaux tirages au hasard), donc les premiers essais et les derniers n'ont pas exactement les mêmes points de test.

### 2.4 Ce qu'on a constaté dans les données (sur 3 scans inspectés)

- **La gencive domine** : 32 % à 69 % des points. Un modèle qui prédirait « gencive » partout aurait déjà une OA d'environ 60 %. **L'OA est donc flatteuse : c'est la mIoU qu'il faut regarder.**
- Certaines classes manquent dans un scan (par exemple des dents de sagesse). C'est plausible, mais je n'ai pas vérifié.

---

## 3. Les deux modèles

Les deux reçoivent `[B, 4096, 3]` et sortent `[B, 4096, 17]` : 17 scores par point.

### 3.1 PointNet (le Student, léger)

**Idée** : appliquer **le même petit réseau à chaque point séparément**, puis résumer tout le nuage par un « maximum » global (une signature de la forme entière), et redonner cette signature à chaque point. Il ne regarde pas explicitement les voisins.

```python
class PointNetSeg(nn.Module):                      # Student (sans T-Net)
    def __init__(self, nc=NUM_CLASSES):
        super().__init__()
        self.f1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.f2 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU())
        self.head = nn.Sequential(nn.Conv1d(1024 + 128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
                                  nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Conv1d(128, nc, 1))
    def forward(self, x):                           # x: [B,N,3] -> logits [B,N,C]
        x = x.transpose(1, 2); l = self.f1(x)                           # caractéristiques locales par point
        g = self.f2(l).max(2, keepdim=True)[0].expand(-1, -1, l.size(2)) # signature globale (max sur les points)
        return self.head(torch.cat([l, g], 1)).transpose(1, 2)           # local + global -> 17 scores par point
```

- `Conv1d(..., 1)` = un petit réseau appliqué point par point, avec les mêmes poids partout.
- `.max(2)` = résumé global : « quelle est la valeur maximale de chaque caractéristique sur tout le nuage ».
- La signature globale est recopiée sur chaque point et concaténée aux caractéristiques locales.
- C'est une version **simplifiée** (sans le T-Net de l'article original).
- **474 129 paramètres.**

### 3.2 DGCNN (le Teacher, plus précis)

**Idée** : pour chaque point, regarder ses **k = 20 voisins les plus proches** et apprendre une caractéristique à partir de la façon dont le point se situe par rapport à eux. Le graphe de voisins est recalculé à chaque couche (« dynamique »). C'est ce qui lui donne le contexte local (courbure, contact entre dents) qui manque à PointNet.

$$x_i' = \max_{j \in \mathcal{N}_k(i)} h_\Theta\big(x_i,\; x_j - x_i\big)$$

En mots : pour le point $i$, on prend chacun de ses $k$ voisins $j$, on forme un couple (position du point, décalage vers le voisin), on le passe dans un petit réseau $h_\Theta$, puis on garde, caractéristique par caractéristique, le **maximum** sur tous les voisins. Ça résume le voisinage en un seul vecteur.

```python
def edge_feat(x, k):                                # x: [B,C,N] -> [B,2C,N,k]
    B, C, N = x.shape
    idx = torch.cdist(x.transpose(1, 2), x.transpose(1, 2)).topk(k, largest=False)[1]   # [B,N,k] les k plus proches
    idx = (idx + torch.arange(B, device=x.device).view(-1, 1, 1) * N).view(-1)
    xt = x.transpose(1, 2).reshape(B * N, C)
    nb = xt[idx].view(B, N, k, C); c = xt.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([nb - c, c], 3).permute(0, 3, 1, 2)   # (voisin - point, point)

class DGCNNSeg(nn.Module):                          # Teacher
    def __init__(self, nc=NUM_CLASSES, k=20):
        super().__init__(); self.k = k
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.e1, self.e2, self.e3 = blk(6, 64), blk(128, 64), blk(128, 64)
        self.mid = nn.Sequential(nn.Conv1d(192, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(nn.Conv1d(1024 + 192, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2),
                                  nn.Dropout(0.5), nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128),
                                  nn.LeakyReLU(0.2), nn.Conv1d(128, nc, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.e1(edge_feat(x, self.k)).max(3)[0]      # couche 1 : voisins dans l'espace 3D
        x2 = self.e2(edge_feat(x1, self.k)).max(3)[0]     # couche 2 : voisins dans l'espace des caractéristiques
        x3 = self.e3(edge_feat(x2, self.k)).max(3)[0]     # couche 3
        l = torch.cat([x1, x2, x3], 1)
        g = self.mid(l).max(2, keepdim=True)[0].expand(-1, -1, l.size(2))
        return self.head(torch.cat([l, g], 1)).transpose(1, 2)
```

- `torch.cdist` calcule toutes les distances entre points : une matrice 4096 × 4096 par scan. **C'est ce qui rend DGCNN coûteux en calcul** (le coût de calcul annoncé dans ton cadrage).
- **562 833 paramètres.**

### 3.3 Une observation importante pour ton projet

**Les deux modèles ont presque le même nombre de paramètres** (474 129 contre 562 833, soit +19 %). DGCNN n'est donc pas « plus gros » mais plus **coûteux en calcul**. Mesures disponibles : à l'entraînement, DGCNN prend ≈ 34 s par époque contre ≈ 4 à 6 s pour PointNet (environ 6 à 8 fois plus). **Ce n'était pas une latence CPU.** Le benchmark CPU a été fait ensuite (partie 8) : sur 1 thread, le Teacher prend environ 433 ms par scan et le Student environ 66 à 68 ms.

---

## 4. Entraînement et mesure de la qualité

### 4.1 Ce que le modèle apprend à minimiser

$$\mathcal{L}_{CE} = -\frac{1}{M}\sum_{i=1}^{M} \log p_{i,\,y_i}$$

En mots : pour chacun des $M$ points, on regarde la probabilité $p_{i,y_i}$ que le modèle donne à la **bonne** classe $y_i$. Plus cette probabilité est proche de 1, plus la perte est petite. On moyenne sur tous les points.

### 4.2 Comment on mesure le résultat

$$\mathrm{OA}=\frac{\text{points bien classés}}{\text{points totaux}}\qquad
\mathrm{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c}\qquad
\mathrm{mIoU}=\frac{1}{|\mathcal{C}|}\sum_{c\in\mathcal{C}}\mathrm{IoU}_c$$

En mots : l'**OA** est le pourcentage de points corrects. L'**IoU** d'une classe $c$ mesure le recouvrement entre les points que le modèle dit « classe $c$ » et ceux qui le sont vraiment : $TP$ = bien trouvés, $FP$ = points pris à tort pour $c$, $FN$ = points de $c$ manqués. La **mIoU** est la moyenne de l'IoU sur les classes présentes dans les données évaluées. Elle punit les erreurs sur les petites classes, alors que l'OA est dominée par la gencive.

```python
@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for p, y in loader:
        pred = model(p.to(dev)).argmax(2).cpu()                       # classe prédite = score le plus élevé
        conf += torch.bincount((y * NUM_CLASSES + pred).view(-1), minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    tp = conf.diag().float(); iou = tp / (conf.sum(0) + conf.sum(1) - tp).float().clamp(min=1)
    present = conf.sum(1) > 0
    return {"OA": (tp.sum() / conf.sum()).item(), "mIoU": iou[present].mean().item()}
```

`conf` est une « matrice de confusion » 17 × 17 : ligne = vraie classe, colonne = classe prédite. Tout (OA, IoU) s'en déduit.

**Attention** : ces métriques sont calculées **par point, sur 4096 points échantillonnés, avec 17 classes partagées**. Elles **ne sont pas** celles du challenge 3DTeethSeg : ne compare pas ces chiffres à son classement.

### 4.3 La boucle d'entraînement (idée)

À chaque époque : le modèle prédit, on calcule la perte, on corrige les poids (optimiseur **Adam** pour `fast.py`, **AdamW** pour `train2.py`, avec un taux d'apprentissage qui décroît en cosinus). Après chaque époque, on mesure sur la **validation**. On garde l'époque qui a la **meilleure validation**, et seulement ensuite on regarde le **test**.

Règle importante respectée dans tout le projet : **on choisit sur la validation, jamais sur le test**.

---

## 5. Les expériences de référence et leurs résultats

Tous les chiffres : même découpage par patient, **test sur 270 scans** de patients jamais vus.

### 5.1 Journal d'expériences (une graine chacune)

| # | Modèle | Recette | Époques | Val mIoU | Test OA | Test mIoU |
|---|---|---|---|---|---|---|
| 1 | PointNet | `fast.py`, bs 16 (1er cache) | 30 | 0,594 | 0,824 | 0,595 |
| 2 | PointNet | `train2.py` : augmentation + perte pondérée | 100 | 0,583 | 0,805 | 0,599 |
| 3 | PointNet | `fast.py`, bs 8 (cache recréé) | 40 | 0,592 | 0,825 | 0,602 |
| 4 | DGCNN | `fast.py`, bs 4 (1er cache) | 20 | 0,672 | 0,869 | 0,665 |
| 5 | DGCNN | `train2.py` : augmentation + perte pondérée | 60 | 0,659 | 0,847 | 0,674 |
| 6 | DGCNN | `fast.py`, bs 4 (cache recréé) | 40 | 0,693 | 0,875 | **0,701** |

### 5.2 Ce qu'on en retient

1. **Le premier essai (30 et 20 époques) était sous-entraîné** : la mIoU montait encore.
2. **Les « améliorations » ont échoué.** Augmentation de données + perte pondérée n'ont rien apporté : PointNet identique à ±0,004, DGCNN moins bon. Hypothèse (non prouvée) : PointNet prédit une position de dent d'après des coordonnées absolues, donc des rotations ou changements d'échelle brouillent le signal ; la pondération fait baisser l'OA sans gain de mIoU. La recette retenue est donc **simple** : cross-entropie, pas d'augmentation, pas de pondération.
3. **Le Student plafonne vers 0,60 de mIoU** quelle que soit la recette ou le nombre d'époques (essais 1, 2, 3 indiscernables avec une graine) : probablement une limite de capacité du modèle.
4. **Le Teacher atteint 0,70 de mIoU / 0,875 d'OA**, soit environ **0,10 de mieux** que le Student. Il y a donc de la marge pour la distillation.
5. **Teacher choisi sur la validation** (0,693 contre 0,659 pour l'autre recette), pas sur le test. Son fichier : `teacher_dgcnn_final.pt` (copie de `dgcnn.pt`).
6. **Le Teacher surapprend un peu** (mIoU de **0,806 sur train**, 0,693 sur val, 0,701 sur test), tout en restant loin de 1,0.
7. Les courbes de validation sont **très bruitées au milieu de l'entraînement** (la mIoU saute de 0,13 à 0,50 d'une époque à l'autre). Elles se stabilisent à la fin, quand le taux d'apprentissage devient petit.

---

## 6. La distillation de connaissances (S3)

### 6.1 L'idée

Le Student apprend à la fois **des vraies étiquettes** et **des probabilités du Teacher**. Les probabilités du Teacher contiennent plus d'information qu'une étiquette (« ce point est la dent 14, mais ressemble un peu à la 15 »).

$$\mathcal{L} = (1-\alpha)\,\mathrm{CE}(z_s, y) + \alpha\, T^2\, \mathrm{KL}\!\big(\mathrm{softmax}(z_t/T)\,\big\|\,\mathrm{softmax}(z_s/T)\big)$$

En mots : on mélange deux erreurs. La première compare la prédiction du Student $z_s$ à la vraie étiquette $y$. La seconde compare la distribution du Student à celle du Teacher $z_t$, toutes deux « adoucies » par la **température** $T$ (plus $T$ est grand, plus les probabilités sont étalées et révèlent les classes proches). Le facteur $T^2$ compense la baisse d'amplitude du gradient due à cet adoucissement. $\alpha$ règle la part du Teacher : 0 = pas de distillation, 1 = Teacher seul.

```python
s = m(p).reshape(-1, C)                                  # logits du Student, un par point
loss = F.cross_entropy(s, y.view(-1), weight=w)          # erreur vs vraies étiquettes
if TL is not None and alpha > 0:
    t = batch[2].to(dev).reshape(-1, C)                  # logits du Teacher (précalculés)
    kl = F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="batchmean") * T * T
    loss = (1 - alpha) * loss + alpha * kl               # mélange des deux erreurs
```

### 6.2 Protocole (pour que la comparaison soit équitable)

- Les logits du Teacher sont **précalculés une fois** sur le cache (échantillon fixe), en mode évaluation. Contrôle : leur mIoU test redonne bien **0,7007 / OA 0,8750**, donc le chargement du Teacher est correct.
- Baseline et Student distillé : **mêmes données, même découpage, mêmes réglages** (`--no_aug --weights none`, 40 époques, bs 8), **3 graines chacun**.
- Commandes :

```bash
python train2.py --model pointnet --epochs 40 --bs 8 --no_aug --weights none --seed 0 --tag base_s0
python train2.py --model pointnet --epochs 40 --bs 8 --no_aug --weights none --seed 0 \
       --kd 4:0.5 --teacher teacher_dgcnn_final.pt --tag kd_T4_a0.5_s0
```

### 6.3 Résultat principal (T = 4, α = 0,5 ; 3 graines)

| Student PointNet | Test OA | Test mIoU |
|---|---|---|
| Baseline (cross-entropie seule) | 0,826 ± 0,001 | **0,608 ± 0,004** |
| Distillé (T = 4, α = 0,5) | 0,822 ± 0,002 | 0,598 ± 0,002 |
| Teacher DGCNN (référence) | 0,875 | 0,701 |

Écart par graine : −0,006, −0,014 et −0,012. La distillation **n'améliore pas** le Student, et le dégrade légèrement dans les 3 graines (écart ≈ 2,5 fois l'écart-type de la baseline). Avec seulement 3 graines, je ne parle pas de preuve statistique.

### 6.4 Petit balayage d'hyperparamètres (graine 0, choisi sur la **validation**)

| Configuration | Val mIoU | (Test mIoU) |
|---|---|---|
| **Baseline** | **0,6114** | 0,6034 |
| KD T = 2, α = 0,3 | 0,6050 | 0,6092 |
| KD T = 2, α = 0,7 | 0,6034 | 0,5943 |
| KD T = 1, α = 0,5 | 0,5985 | 0,6056 |
| KD T = 4, α = 0,5 | 0,5946 | 0,5976 |
| KD T = 4, α = 0,9 | 0,5870 | 0,5849 |

Règle fixée à l'avance : on ne retient un couple que s'il dépasse la baseline en **validation** d'au moins 0,005 (le bruit entre graines est de cet ordre). **Aucun** couple n'y arrive : tous sont **en dessous** de la baseline en validation (0,587 à 0,605 contre 0,611). Le résultat négatif est donc confirmé.

Transparence : le couple T = 2, α = 0,3 est légèrement au-dessus de la baseline *sur le test* (+0,006), mais c'est dans le bruit (la baseline varie de 0,603 à 0,614 selon la graine) et il n'est pas meilleur en validation. Je ne le retiens pas : choisir d'après le test serait de la triche involontaire.

### 6.5 Pourquoi ça ne marche pas ? (hypothèses, pas des preuves)

- **Capacité du Student** : il plafonne à ≈ 0,60 quelle que soit la recette. Les « leçons » du Teacher ne lui servent peut-être pas s'il ne peut pas les représenter.
- **Le Teacher a surappris** : sa mIoU est de 0,806 sur ses scans d'entraînement contre 0,70 sur des scans neufs. Ses réponses sur les données d'entraînement sont plus sûres et plus justes que ce qu'il fera en pratique, donc moins informatives. Ce point est **mesuré** ; le lien de causalité avec l'échec ne l'est pas.
- **Réglages limités** : seulement 6 couples (T, α) essayés, sur une graine.

### 6.6 Ce qui n'est pas fait

Au moment où ce guide initial a été écrit, la distillation des **couches intermédiaires** (prévue dans le cadrage) n'avait pas été essayée. Elle l'a été ensuite sur une graine : voir la partie 11 (résultat non confirmé).

---

---

## 7. La compression (semaine 4) : les trois outils

### 7.1 Pourquoi compresser, et ce qu'on mesure

Le but est un modèle **petit et rapide** qui perde peu de précision. On suit trois axes :
- la **taille** du fichier (Mo) ;
- la **latence** : millisecondes pour traiter un nuage de 4096 points sur CPU, 1 thread ;
- la **précision** : mIoU.

C'est toujours un **compromis** : gagner sur un axe coûte presque toujours sur un autre.

### 7.2 Première étape : fusionner Conv1d et BatchNorm

En inférence, la BatchNorm est une opération figée (un décalage et un étirement par canal) qui suit une couche linéaire. On peut donc fusionner les deux en une seule couche, sans changer le résultat.

$$y = \gamma \,\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}} + \beta \qquad W' = \frac{\gamma}{\sqrt{\sigma^2+\varepsilon}}\, W \qquad b' = \frac{\gamma\,(b-\mu)}{\sqrt{\sigma^2+\varepsilon}} + \beta$$

En mots : on multiplie chaque ligne de $W$ par le facteur d'échelle de la BatchNorm, et on absorbe le décalage dans le biais. Le résultat est identique avec une couche de moins.

Une `Conv1d` à noyau 1 fait exactement ce que fait une couche `Linear` appliquée à chaque point : le réseau fusionné est donc un enchaînement de `Linear` (l1, l2, l3, h1, h2, out).

```python
def fuse(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    lin = nn.Linear(conv.in_channels, conv.out_channels)
    lin.weight.data = conv.weight.data[:, :, 0] * s[:, None]
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    lin.bias.data = (b - bn.running_mean) * s + bn.bias
    return lin
```

**Vérification mesurée** : l'écart maximal entre le modèle d'origine et le modèle fusionné est de **1,9 × 10⁻⁵** (bruit d'arrondi). Après fusion, le Student a **470 929** paramètres (les 3 200 paramètres de BatchNorm ont disparu). Son score est inchangé : val mIoU **0,6114**, OA 0,831 ; test mIoU 0,6034.

### 7.3 Pruning non structuré : mettre des poids à zéro

**Principe** : on met à zéro les poids de plus petite valeur absolue (L1 global, `torch.nn.utils.prune`), puis on réentraîne quelques époques.

**Résultats mesurés** (première version : 5 époques, lr constant 3×10⁻⁴, graine de base 0) :

| Modèle | Val mIoU | Poids non nuls | Taille | Latence CPU |
|---|---|---|---|---|
| Student fusionné | 0,611 | 470 929 | 1,89 Mo | 66–69 ms |
| Pruning 50 % | 0,585 (test 0,580) | 236 273 | 1,89 Mo | 67 ms |
| Pruning 70 % | 0,515 | 142 411 | 1,89 Mo | 69 ms |
| Pruning 90 % | 0,348 | 48 548 | 1,89 Mo | 68 ms |

**Leçon** : la taille et la latence **ne changent pas**. Les zéros restent stockés, et un CPU dense multiplie les zéros comme les autres nombres. Seule la précision baisse. Pour gagner en vitesse, il faut supprimer des **neurones entiers** (partie 8). Nous n'avons pas essayé de noyaux de calcul creux.

**Deux réserves** :
- le 90 % n'avait pas convergé (la mIoU montait encore à l'époque 5) : 0,348 est un plancher ;
- cette recette de réentraînement abîmait déjà le modèle (voir 8.3) : la part de perte due au pruning n'est **pas isolée**.

### 7.4 Quantification INT8 dynamique

**Principe** : on stocke chaque poids sur 1 octet au lieu de 4.

$$q=\mathrm{round}\!\left(\frac{x}{s}\right)+z \qquad \hat{x}=s\,(q-z)$$

En mots : $s$ est le pas (l'échelle) et $z$ le décalage. On arrondit, donc on perd un peu d'information : c'est cette erreur d'arrondi qui coûte de la précision. « Dynamique » signifie que les poids sont quantifiés une fois et les activations à la volée, pendant l'inférence.

```python
q = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

**Résultats mesurés** (première exécution) : taille 1,89 → **0,48 Mo** (÷ 3,9) ; val mIoU 0,611 → 0,602 ; test 0,603 → 0,596 ; latence 68,8 → 60,7 ms. Le pruning 50 % + INT8 donne val 0,576, test 0,574, sans gain de latence supplémentaire.

**Teacher (référence, même script)** : val mIoU 0,693, taille 2,28 Mo, latence **433 ms**.

---

## 8. Le pruning structuré, les contrôles et les résultats finaux

### 8.1 Où se trouve le coût de calcul

Une couche `Linear(in, out)` coûte $in \times out$ multiplications-accumulations (MAC) par point.

| Couche | Taille | MAC/point | Part |
|---|---|---|---|
| l1 | 3 → 64 | 192 | 0,04 % |
| l2 | 64 → 128 | 8 192 | 1,7 % |
| **l3** | 128 → 1024 | 131 072 | **28 %** |
| **h1** | 1152 → 256 | 294 912 | **63 %** |
| h2 | 256 → 128 | 32 768 | 7 % |
| out | 128 → 17 | 2 176 | 0,5 % |

Deux couches pèsent **91 %** du calcul : c'est là qu'on coupe.

**Schéma du Student** :
```
point (x,y,z) → l1 (64) → l2 (128)  ──► caractéristiques locales (128)
                                  └─► l3 (A neurones) → max sur tous les points → signature globale (A)
concaténation : 128 + A  →  h1 (B neurones) → h2 (128) → out (17 classes)
```

**Notation « A/B »** : A = nombre de neurones de l3, B = nombre de neurones de h1. Le Student complet est **1024/256**. Les autres configurations testées : **512/256**, **256/128**, **128/64**.

### 8.2 Le principe du pruning structuré

On garde les $k$ neurones $j$ de plus grande importance :

$$\mathrm{importance}_j=\sum_i |W_{j,i}|$$

En mots : pour chaque neurone, on additionne la valeur absolue de ses poids d'entrée. Un neurone aux poids minuscules produit une sortie faible et contribue peu. C'est une **heuristique** : nous ne l'avons pas comparée à d'autres critères.

Réduire l3 réduit aussi l'entrée de h1 ($128+A$) : la coupe **se propage** et les matrices deviennent réellement plus petites.

| Config | Paramètres | MMAC/point | Taille fp32 | Taille INT8 |
|---|---|---|---|---|
| 1024/256 (complet) | 470 929 | 0,469 | 1,89 Mo | 0,48 Mo |
| 512/256 | 273 809 | 0,273 | 1,10 Mo | 0,28 Mo |
| 256/128 | 109 585 | 0,109 | 0,44 Mo | 0,12 Mo |
| 128/64 | 52 049 | 0,052 | 0,21 Mo | 0,06 Mo |

**Juste après la coupe, sans réentraînement**, la mIoU de validation tombe à **0,026** pour les trois configurations réduites : le modèle s'effondre (probablement une prédiction « gencive » partout, hypothèse non vérifiée). Toute la récupération vient du réentraînement. Dans le cas trivial (rien coupé), le code redonne bien 0,6114 : l'indexation est correcte dans ce cas.

### 8.3 Le témoin : sans lui, on aurait mal interprété

Un **témoin** réentraîne le Student complet **sans rien supprimer**, avec la même recette. Il mesure ce que coûte le réentraînement lui-même.

| Recette (témoin 1024/256) | Val mIoU | Test mIoU |
|---|---|---|
| Student de base, non réentraîné | 0,611 | 0,603 |
| lr 3×10⁻⁴, 8 époques (ancienne recette, 3 graines de fine-tuning) | 0,592 ± 0,002 | 0,584 ± 0,003 |
| lr 10⁻⁴, 20 époques (recette retenue, 2 graines) | **0,613** ± 0,000 | 0,606 |

**Leçon** : l'ancienne recette abîmait déjà le modèle de ≈ 0,02. Sans témoin, cette perte aurait été attribuée à la coupe. Avec la recette retenue, le témoin revient au niveau du modèle de base. **Limite** : deux paramètres (lr et nombre d'époques) ont changé à la fois, on ne sait pas lequel corrige.

### 8.4 Pruning structuré, ancienne recette, puis nouvelle recette (CE vs distillation)

| Config | Ancienne recette (val) | Nouvelle recette CE (val / test) | Nouvelle recette + KD logits (val / test) |
|---|---|---|---|
| 1024/256 | 0,592 | 0,613 / 0,606 | 0,602 / 0,597 |
| 512/256 | 0,545 | 0,564 / 0,560 | 0,560 / 0,555 |
| 256/128 | 0,460 | 0,474 / 0,479 | 0,462 / 0,465 |
| 128/64 | 0,417 | 0,420 / 0,418 | 0,418 / 0,416 |

- La nouvelle recette améliore les largeurs moyennes (+0,02 sur 512/256, +0,014 sur 256/128) ; à 128/64 les courbes plafonnent à environ 0,42 avec les deux recettes (limite de capacité possible, **non testée**).
- **La distillation des logits pendant le réentraînement n'aide pas** : KD − CE = −0,011, −0,005, −0,012, −0,002 (val, 2 graines, un seul réglage T = 4, α = 0,5). Cohérent avec la partie 6.
- Contrôle effectué : les logits du Teacher en cache redonnent val 0,6930 / test 0,7007, donc c'est bien le Teacher final.

### 8.5 Résultats finaux (3 entraînements de base, recette CE lr 10⁻⁴, 20 époques)

Moyenne ± écart-type de **population** (n = 3 ; l'écart-type d'échantillon serait ≈ 22 % plus grand). Mesures de latence : médiane de 50 passages, 1 thread, une seule série (≈ ±10 %).

| Modèle | Val mIoU | Test mIoU | Taille | Latence | MMAC/pt |
|---|---|---|---|---|---|
| Teacher DGCNN (1 graine, mesure antérieure) | 0,693 | 0,701 | 2,28 Mo | 433 ms | – |
| 1024/256 fp32 | 0,607 ± 0,005 | **0,611 ± 0,004** | 1,89 Mo | 68,0 ms | 0,469 |
| 1024/256 INT8 | 0,598 ± 0,005 | 0,601 ± 0,003 | 0,48 Mo | 46,9 ms | 0,469 |
| 512/256 fp32 | 0,564 ± 0,005 | 0,563 ± 0,004 | 1,10 Mo | 26,6 ms | 0,273 |
| 512/256 INT8 | 0,558 ± 0,006 | 0,556 ± 0,003 | 0,28 Mo | 18,6 ms | 0,273 |
| 256/128 fp32 | 0,487 ± 0,011 | 0,490 ± 0,010 | 0,44 Mo | 10,7 ms | 0,109 |
| 256/128 INT8 | 0,481 ± 0,011 | 0,484 ± 0,010 | 0,12 Mo | 8,2 ms | 0,109 |
| 128/64 fp32 | 0,427 ± 0,007 | 0,423 ± 0,007 | 0,21 Mo | 5,2 ms | 0,052 |
| 128/64 INT8 | 0,428 ± 0,007 | 0,423 ± 0,006 | 0,06 Mo | 4,6 ms | 0,052 |

**Coût de l'INT8 sur le test, par entraînement de base** (INT8 − fp32) : 1024/256 : −0,008, −0,009, −0,012 ; 512/256 : −0,004, −0,006, −0,009 ; 256/128 : −0,006, −0,006, −0,007 ; 128/64 : +0,002, −0,001, −0,002 (non mesurable).

**Exemple de lecture** : 512/256 INT8 est ≈ **23 fois plus rapide** et **8 fois plus petit** que le Teacher, pour **0,145 de mIoU en moins**. Attention : le Teacher n'est pas compressé et son architecture est différente (graphe de voisins), donc cet écart mélange architecture et compression.

### 8.6 Le front de Pareto

Un modèle A **domine** B si A est meilleur ou égal sur tous les critères et strictement meilleur sur au moins un. Un modèle **dominé** est donc moins bon qu'un autre. Les modèles **non dominés** forment le **front de Pareto** : chacun est un compromis valable. (Exemple de contrôle : A = 0,70 / 400 ms, B = 0,60 / 60 ms, C = 0,55 / 70 ms : A et B sont sur le front, C est dominé par B.)

Ici, **tous les points sont sur le front** au sens strict : plus on réduit, plus on gagne en vitesse et en taille et plus on perd en précision. Il n'y a pas de « coude » net qui désignerait un meilleur choix : cela dépend de la contrainte. Une nuance à 128/64 : l'INT8 égale le fp32 en précision (écart ≤ 0,0002) tout en étant plus petit et plus rapide, donc il le domine à l'erreur de mesure près.

Figure : `pareto_clean.png` (latence en échelle log en abscisse, mIoU test en ordonnée, ronds = fp32, carrés = INT8).

### 8.7 La mémoire vive d'inférence

Première mesure **invalide** : `forward_delta_MB = 0` et RSS identique (1 433,8 Mo) pour tous les modèles. Cause : le pic mémoire était atteint au chargement (PyTorch, CUDA, trimesh) et l'inférence ne le dépassait jamais ; l'erreur venait de ma méthode, pas du modèle.

Mesure corrigée (`ram_bench.py` : processus neuf, poids aléatoires, échantillonnage du RSS courant, 1 thread, 1 nuage de 4096 points) :

| Config | Surcoût fp32 | Surcoût INT8 |
|---|---|---|
| 1024/256 | 37,4 Mo | 37,8 Mo |
| 512/256 | 21,9 Mo | 23,2 Mo |
| 256/128 | 13,8 Mo | 12,6 Mo |
| 128/64 | 13,7 Mo | 12,6 Mo |

- Le pruning structuré réduit la RAM d'inférence (37 → 14 Mo).
- L'INT8 dynamique **ne la réduit pas**. Explication probable (**hypothèse non mesurée séparément**) : les activations (tableaux intermédiaires de 4096 points) dominent, alors que les poids ne pèsent que ≈ 2 Mo.
- Le plateau à ≈ 14 Mo (256/128 et 128/64) n'est pas expliqué.
- Le RSS de base (≈ 407 Mo) est celui de Python et PyTorch, pas du modèle.
- Limites : une mesure par config ; des pics très brefs ont pu être manqués (valeurs basses).

---

## 9. La présentation Beamer (semaines 3 et 4)

**Contenu (17 diapositives, en français)** : statut ; rappel S2 ; distillation (principe, résultat, hypothèses) ; coût de calcul ; pruning non structuré ; pruning structuré ; contrôles ; INT8 ; protocole de mesure ; résultats finaux ; Pareto ; RAM ; limites ; conclusion.

**Fichiers** : `presentation_s3_s4.tex`, `presentation_s3_s4.pdf`, `pareto_clean.png` (à garder dans le même dossier). Compilation recommandée : `xelatex presentation_s3_s4.tex` (deux fois). La voie `pdflatex` (babel) n'a pas été testée.

**Vérification critique effectuée** (et corrections appliquées) :
- Les 8 lignes du tableau final ont été comparées au fichier `final_results.json` : correspondance exacte.
- Corrigé : renvoi à une diapositive inexistante ; pruning non structuré présenté sans réserve (recette confondante) ; comparaison avec un Teacher non compressé présentée comme un gain pur ; écart-type de population non signalé ; « aucun modèle n'en domine un autre » trop fort à 128/64 ; explication RAM présentée comme un fait.
- **Reste à faire** : ajouter le modèle du CPU (`lscpu | grep "Model name"`) ; un `TODO` est dans le `.tex`.

**Speech** : trois versions ont été produites (long, précis et clair, court de ≈ 9–10 min).

---

## 10. Les courbes d'apprentissage (loss train et validation)

Les scripts d'entraînement d'origine n'enregistraient que la loss d'entraînement et la mIoU de validation. Deux scripts ont été fournis : `plot_logs.py` (lit `log_base_s0.txt`) et `curves_loss.py` (réentraîne PointNet en enregistrant aussi la loss de validation, ≈ 4 min).

**Ce que les courbes montrent** :
1. **Pas de surapprentissage classique** : la loss de validation ne remonte jamais ; la mIoU est maximale à l'époque 40.
2. **Un écart de généralisation** : loss ≈ 0,27 sur des scans d'entraînement évalués en mode évaluation, ≈ 0,44 en validation. Cela nuance l'hypothèse « le Student est limité par sa seule capacité ».
3. **Instabilité du milieu d'entraînement liée au mode évaluation, pas aux données** : les courbes « train (mode eval) » et « validation » ont des pics synchrones, alors que la loss d'entraînement en mode entraînement est lisse. Hypothèse (non testée) : les statistiques de BatchNorm sont décalées à fort learning rate ; les pics s'atténuent quand le learning rate décroît.
4. Choisir le meilleur checkpoint en milieu d'entraînement serait bruité ; ici le meilleur est en fin d'entraînement.

**Avertissement** : `curves_loss.py` est une **réexécution** (ordre des données différent) : sa mIoU de validation finale est ≈ 0,60 contre 0,611 dans le log d'origine. Cet écart de ≈ 0,01 est de la taille des effets discutés (par exemple l'INT8). À utiliser pour la **forme** des courbes, pas pour remplacer les résultats.

---

## 11. Distillation sur les couches intermédiaires (feat_kd.py)

### 11.1 Quel type de distillation avait-on fait avant ?

**Distillation de connaissances sur les logits** (Hinton), **hors ligne** (Teacher entraîné d'abord, figé, logits calculés une fois et stockés en float16), **par point** (segmentation dense), avec un Teacher et un Student de familles différentes (DGCNN, PointNet). Perte : $\mathcal{L}=(1-\alpha)\mathrm{CE}+\alpha T^2\mathrm{KL}$, réglages principaux $T=4$, $\alpha=0{,}5$. **Pas** de couches intermédiaires, **pas** de distillation en ligne, **pas** d'auto-distillation.

### 11.2 Le principe de la distillation des couches intermédiaires

Analogie : jusqu'ici le Student copiait les **réponses finales** ; ici il copie aussi le **raisonnement** (représentations internes). On compare les caractéristiques par point à des étages qui se correspondent :

| Étage | Teacher DGCNN | Student PointNet |
|---|---|---|
| Local | 192 canaux par point (avec voisinage) | 128 canaux par point |
| Couche cachée | 256 canaux par point (avant logits) | 256 canaux par point |
| Global | vecteur de 1024 par scan | vecteur de 1024 par scan |

$$\mathcal{L}=\mathrm{CE}+\lambda\sum_k\big(1-\cos(\text{projecteur}_k(\text{Student}_k),\,\text{Teacher}_k)\big)$$

En mots : à l'erreur de classification normale, on ajoute, pour chaque étage choisi, un terme qui rapproche la **direction** des vecteurs du Student (après un petit adaptateur linéaire, jeté après l'entraînement) de ceux du Teacher. Le cosinus ignore l'échelle des valeurs. $\lambda$ pondère ce terme (1,0 par défaut, non réglé).

Le Teacher voit les voisins de chaque point, pas le Student : imiter ces représentations est plus dur que d'imiter la réponse finale, et rien ne garantit un gain. Le Teacher est **figé et relancé à chaque lot** (en ligne), ce qui rend l'entraînement ≈ 6 fois plus long (mesuré : 166 s pour `base`, ≈ 1020 s pour `hid` et `all`) sans changer le coût d'inférence du modèle final.

### 11.3 Les quatre variantes et la règle

`base` (CE seul, témoin dans la même boucle), `hid` (couche cachée), `all` (local + cachée + globale), `all_logit` (`all` + logits). **Règle fixée à l'avance** : retenir une variante seulement si Δval ≥ +0,005 par rapport à `base`, sur les mêmes graines.

### 11.4 Résultats (graine 0 seulement, 40 époques)

| Variante | Val mIoU | Test mIoU | Δ val vs base |
|---|---|---|---|
| base | 0,6010 | 0,6068 | – |
| hid | 0,6106 | 0,6081 | **+0,0096** |
| all | 0,6067 | 0,6102 | **+0,0057** |
| all_logit | 0,5973 | 0,5946 | −0,0037 |

**Lecture honnête** :
- `hid` et `all` franchissent la règle sur **une seule graine**, ce qui est insuffisant.
- Le `base` de cette boucle (0,6010) est **inférieur** au Student de base d'origine (0,6114), avec la même graine et la même recette : le bruit lié à l'ordre des données est de ≈ 0,010, de la même taille que le « gain ». `hid` (0,6106) ne dépasse d'ailleurs pas le base d'origine (0,6114).
- Sur le test, les écarts sont minuscules (+0,0013 pour `hid`, +0,0034 pour `all`).
- `all_logit` est le pire (−0,012 en test) : ajouter les logits n'aide pas.
- Le meilleur checkpoint est à l'époque 39–40 pour les quatre variantes (cosinus jusqu'à la fin).
- Les valeurs de loss ne sont **pas comparables** entre variantes (elles incluent des termes différents).

### 11.5 Suite prévue

Relancer `base`, `hid`, `all` sur les graines 1 et 2 (≈ 70–80 min), en sauvegardant d'abord `feat_kd_results.json` (le script l'écrase), puis fusionner. **Critère** : un Δval ≥ +0,005 dans **les trois** graines. Sinon : « pas de gain démontré, tendance non confirmée ». Si `hid` se confirme, l'appliquer au réentraînement des petits modèles (512/256, 256/128).


---

## 12. Mini-cours : le vocabulaire et les questions de compréhension

### 12.1 Ce que veulent dire les noms (abréviations)

| Nom | Sens |
|---|---|
| `s0`, `s1`, `s2` | **seed** (graine) 0, 1, 2 : le nombre qui fixe le hasard d'un entraînement. `base_s0_best.pt` = Student de base, graine 0, meilleur checkpoint sur la validation. On répète avec plusieurs graines pour distinguer un vrai effet du hasard. |
| `e1`, `e2`, `e3` | Teacher DGCNN : trois blocs d'**edge convolution** (convolution sur les arêtes d'un graphe de 20 plus proches voisins), 64 canaux chacun. Les trois sorties sont concaténées (3 × 64 = 192 canaux « locaux »). |
| `f1`, `f2` | Student : blocs de **f**eatures 1 (3 → 64 → 128) et 2 (128 → 1024). |
| `l1`, `l2`, `l3`, `h1`, `h2`, `out` | Student **fusionné** : couches *l*ocales (l) et couches de la *h*ead (h). |
| `mid` | Teacher : couche 192 → 1024 avant le max global. |
| `head` | Partie finale qui produit les 17 scores par point. |
| `T_`, `S` (dans `feat_kd.py`) | Teacher, Student. |
| **A/B** (1024/256, 512/256…) | Nombre de neurones de **l3** / nombre de neurones de **h1**. 1024/256 = Student complet. |
| **fp32 / INT8** | Poids sur 4 octets (flottants) / sur 1 octet (entiers). |
| **MAC** | Multiplication-accumulation : unité de coût de calcul. |
| **Logits** | Scores bruts par classe, avant les probabilités. |
| **Témoin** | Le même réentraînement sans rien supprimer, pour mesurer ce que coûte la procédure elle-même. |

### 12.2 Les questions posées et les réponses à retenir

1. **« Pourquoi le pruning non structuré n'accélère-t-il pas le CPU ? »** Parce qu'un CPU effectue des multiplications de matrices **denses** : il multiplie les zéros comme tout autre nombre. Il faut des matrices réellement plus petites.
2. **« Qu'est-ce qu'un modèle dominé ? »** Un modèle **moins bon** qu'un autre (plus précis ou égal, ET plus rapide ou égal, avec au moins une inégalité stricte). Les non dominés forment le front de Pareto.
3. **« Pourquoi choisit-on sur la validation et pas sur le test ? »** Chaque choix « le meilleur » calibre un peu le modèle sur l'ensemble utilisé. Si on choisit sur le test, il ne reste plus d'ensemble neuf pour mesurer la vraie performance. Le léger écart val/test (0,611 / 0,603) est normal.
4. **« Le pruning structuré ne marche pas ? »** Si : il donne de vrais gains de latence (÷ 13), de taille et de mémoire. Ce qui coûte, c'est la **précision**.
5. **« Un résultat négatif, c'est quoi ? »** Une expérience bien faite dont la réponse est « non, cela n'améliore pas ». Ce n'est pas une erreur, c'est un résultat valide, à rapporter tel quel. On peut dire « pas de gain avec les réglages testés », pas « ne marche jamais ».
6. **Lecture du graphique de Pareto** : plus à gauche = plus rapide ; plus haut = plus précis ; le meilleur serait en haut à gauche, il n'existe pas. On choisit selon la contrainte (par exemple : < 20 ms → 512/256 INT8 ; précision ≥ 0,60 → 1024/256 INT8).
7. **Lecture des courbes de loss** : train et validation qui descendent ensemble = le modèle apprend ; train qui descend et validation qui remonte = surapprentissage ; un grand écart = généralisation limitée. La loss mesure la confiance des probabilités, la mIoU la justesse des classes : leurs optimums peuvent différer.

---

## 13. Carte des fichiers (dans `IA et APPLICATIONS/`)

### 13.1 Données et entraînement (parties 1 à 6)

| Fichier | Rôle |
|---|---|
| `teeth_kd.py` | Lecture des scans, 17 classes, échantillonnage à 4096 points, découpage par patient, modèles PointNet et DGCNN, métriques. |
| `fast.py` | Construit `cache_4096.npz` (`--mode cache`), puis entraînement rapide depuis le cache. |
| `train2.py` | Entraînement complet (meilleur checkpoint sur la validation, graines, augmentation, pondération, distillation `--kd`). |
| `summary.py` | Moyenne ± écart-type baseline contre distillation. |
| `kd.py` | **Obsolète, jamais lancé.** |
| `cache_4096.npz` | Les 1800 scans en 4096 points. **Ne pas régénérer.** |
| `teacher_dgcnn_final.pt` | Le Teacher retenu (copie de `dgcnn.pt`). |
| `teacher_logits_teacher_dgcnn_final.npy` | Logits du Teacher en cache (1800 × 4096 × 17, float16). |
| `base_s0/1/2_best.pt`, `*_result.json`, `log_*.txt` | Checkpoints, résultats et journaux des entraînements de base. |

### 13.2 Compression et mesures (parties 7 à 11)

| Fichier | Rôle |
|---|---|
| `compress.py` | Fusion BN, pruning non structuré (50/70/90 %), INT8 dynamique, benchmark initial avec le Teacher. |
| `structured_prune.py` | Pruning structuré (4 largeurs), 3 graines de fine-tuning, INT8, latence, taille ; sa mesure RAM était invalide. |
| `ft_kd.py` | Réentraînement doux (lr 10⁻⁴, 20 époques) : CE contre CE + distillation des logits. |
| `ram_bench.py` | Mesure corrigée du surcoût mémoire d'inférence. |
| `final_runs.py` | Expérience finale sur 3 entraînements de base : `final_results.json`, `pareto.png`. |
| `plot_logs.py`, `curves_loss.py` | Courbes depuis les logs / réentraînement avec loss de validation. |
| `feat_kd.py` | Distillation des couches intermédiaires (4 variantes) : `feat_kd_results.json`. |
| `presentation_s3_s4.tex/.pdf`, `pareto_clean.png` | Présentation Beamer. |
| Guides `.md` | `teeth3ds_dataset_problemes_solutions.md`, `guide_complet_projet_1_5.md`, ce document. |

---

## 14. Tous les problèmes rencontrés et leurs solutions

| # | Problème | Solution |
|---|---|---|
| 1 | Le cadrage S1 dit que le téléchargement a été testé, alors que le dataset n'était pas téléchargé | Télécharger, et ne pas répéter cette affirmation sans vérification |
| 2 | Archives dans des archives (0 `.obj` après la 1re extraction) | Extraire les zips internes |
| 3 | PART3 tronquée | Re-téléchargée, testée avec `unzip -tq` |
| 4 | Faux « CASSE » sur PART5 : chemin avec espace coupé par ma boucle | Guillemets autour du chemin |
| 5 | Part 7 sans `.json` (100 scans) | Ignorée automatiquement |
| 6 | Fuite de données dans le split officiel (382 patients) | Découpage par patient |
| 7 | Listes 3DTeethLand (340 scans) inadaptées | Non utilisées |
| 8 | Scans très denses (84 000 à 186 000 sommets) | Échantillon fixe de 4096 points |
| 9 | Gencive dominante (32 à 69 %) : l'OA est trompeuse | On suit la mIoU |
| 10 | Bug d'affichage (`%g`) dans la première version de `teeth_kd.py` | Corrigé |
| 11 | Lecture des `.obj` trop lente | Cache `cache_4096.npz` |
| 12 | Cache régénéré = points de test changés | Ne plus régénérer |
| 13 | Recette « améliorée » (augmentation + pondération) pas meilleure | Recette simple retenue |
| 14 | Distillation des logits sans gain | Résultat négatif documenté (balayage sur la validation) |
| 15 | **Pas de torch dans l'environnement de l'assistant** (installation impossible) | Tous les scripts sont exécutés sur ta machine ; ils sont livrés non testés par l'assistant (syntaxe vérifiée seulement) |
| 16 | Fine-tuning du pruning sur CPU : bloqué ≈ 10 min | Fine-tuning et évaluations de validation sur GPU, mesures de latence sur CPU |
| 17 | Mesure de RAM invalide (pic mémoire atteint au chargement) | `ram_bench.py` : processus neuf, RSS courant échantillonné |
| 18 | **Recette de réentraînement qui abîmait le modèle** (témoin 0,611 → 0,592) | lr 10⁻⁴ et 20 époques (témoin 0,613) ; le témoin est désormais systématique |
| 19 | Pruning non structuré évalué avec l'ancienne recette | Réserve notée ; non rejoué |
| 20 | Compilation Beamer : paquet de français absent chez l'assistant | XeLaTeX + polyglossia (voie `pdflatex`/babel non testée) |
| 21 | Présentation : plusieurs formulations trop fortes | Corrigées après revue (voir partie 9) |
| 22 | Distillation des couches : le `base` de la nouvelle boucle diffère du base d'origine de ≈ 0,010 | Une graine ne suffit pas ; relancer sur 3 graines |

---

## 15. Ce qu'on peut affirmer, ce qu'on ne peut pas

### Établi (mesuré)
- Le jeu de données est caractérisé et ses pièges identifiés (fuite par patient, archives, part 7 sans annotations) ; l'alignement longueur points/labels est vérifié sur les 1800 scans.
- Teacher DGCNN : mIoU test 0,701 ; Student PointNet : 0,608 ± 0,004 (3 graines).
- La distillation des **logits** n'améliore pas le Student (0,608 contre 0,598), ni pendant le réentraînement des Students réduits (réglage T = 4, α = 0,5).
- Le pruning **non structuré** ne réduit ni la taille ni la latence sur CPU dense.
- Le pruning **structuré** réduit la latence (66 → 5 ms), la taille (1,89 → 0,21 Mo) et la RAM d'inférence (37 → 14 Mo), au prix de la précision (test 0,611 → 0,423).
- L'INT8 dynamique divise la taille par ≈ 4 pour ≈ 0,006 à 0,012 de mIoU (modèles 1024/256 à 256/128) ; non mesurable à 128/64 ; il ne réduit pas la RAM mesurée.
- Le **témoin** a montré que l'ancienne recette coûtait ≈ 0,02.

### Non établi (hypothèses ou non fait)
- La **cause** de l'échec de la distillation (capacité du Student, surapprentissage du Teacher).
- Le plafond de précision des petits modèles (sous-entraînement ? capacité ?).
- L'efficacité de la distillation des **couches intermédiaires** (1 graine, signal non confirmé).
- Le rôle du critère L1, de la quantification statique, de noyaux creux.
- L'explication RAM (activations dominantes), le plateau à 14 Mo.
- Comparabilité avec le challenge 3DTeethSeg (protocole différent) : **non comparable**.
- Les labels n'ont pas été vérifiés visuellement.

### Limites à mentionner
Un seul découpage ; Teacher sur une graine ; latences en une série (≈ ±10 %), sur un seul CPU portable (modèle à préciser) et 1 thread ; écart-type de population sur n = 3 ; petits modèles plus instables (±0,010 à 256/128) ; hyperparamètres de distillation peu explorés ; Students réduits peut-être sous-entraînés.

---

## 16. Commandes utilisées (dans le dossier du projet)

```bash
cd "/home/abayad/Documents/IA et APPLICATIONS"
python fast.py --mode cache                                                      # cache 4096 points (ne pas refaire)
python train2.py --model pointnet --epochs 40 --bs 8 --no_aug --weights none --seed 0 --tag base_s0
python train2.py --model pointnet --epochs 40 --bs 8 --no_aug --weights none --seed 0 \
       --kd 4:0.5 --teacher teacher_dgcnn_final.pt --tag kd_T4_a0.5_s0           # distillation des logits
/usr/bin/python compress.py --ckpt base_s0_best.pt --teacher teacher_dgcnn_final.pt
/usr/bin/python structured_prune.py --ckpt base_s0_best.pt
/usr/bin/python ft_kd.py
/usr/bin/python ram_bench.py
/usr/bin/python final_runs.py                                                    # tableau final + pareto.png + final_results.json
/usr/bin/python plot_logs.py log_base_s0.txt log_kd_s0.txt
/usr/bin/python curves_loss.py                                                   # courbes loss train/val (≈ 4 min)
/usr/bin/python feat_kd.py                                                       # distillation des couches, graine 0
# suite prévue (sauvegarder d'abord le JSON, car le script l'écrase) :
cp feat_kd_results.json feat_kd_results_s0.json
/usr/bin/python feat_kd.py --seeds 1,2 --variants base,hid,all
cp feat_kd_results.json feat_kd_results_s12.json
xelatex presentation_s3_s4.tex                                                   # (deux fois) présentation
lscpu | grep "Model name"                                                        # à ajouter à la diapositive 12
```

---

## 17. Ce qui reste à faire

1. **Distillation des couches intermédiaires** : lancer les graines 1 et 2 (`base`, `hid`, `all`), fusionner, appliquer la règle (Δval ≥ +0,005 dans les trois graines). Si `hid` se confirme, l'appliquer au réentraînement des petits modèles.
2. **Rapport final (semaine 5)** : contexte, méthodes, résultats, front de Pareto, limites. À rédiger en LaTeX, en français.
3. **Présentation** : ajouter le modèle du CPU ; éventuellement alléger les diapositives denses et ajouter une diapositive de synthèse avec une recommandation de modèle ; intégrer, si pertinent, les courbes d'apprentissage et la distillation des couches.
4. **Optionnel** : quantification statique, entraînement plus long des petits modèles, autres critères de coupe, rejouer le pruning non structuré avec la recette finale, mesures de latence répétées, Teacher sur plusieurs graines.

**Dernier état au moment de ce récapitulatif** : la distillation des couches intermédiaires sur la graine 0 donne `hid` +0,0096 et `all` +0,0057 en validation par rapport au témoin, mais ce résultat est dans le bruit d'une seule graine (≈ 0,010) : **non confirmé**.
