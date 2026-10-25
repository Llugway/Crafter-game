# Le projet à l'état 0 : ce que fait le code, en détail

Ce document décrit le code tel qu'il était en 2022, avec seulement les correctifs de portabilité
du README. Il sert de référence : chaque amélioration future sera décrite par rapport à lui.

---

## 1. Le jeu : Crafter

Crafter (Hafner, 2021) est un Minecraft 2D simplifié, conçu pour mesurer l'éventail des
capacités d'un agent : exploration, survie, planification sur le long terme.

- **Monde** : 64×64 cases, regénéré à chaque partie (prairies, forêts, lacs, montagnes, grottes,
  lave). Jour et nuit alternent ; la nuit, l'image s'assombrit et les zombies sont plus nombreux.
- **Observation** : une image RGB 64×64. On y voit une fenêtre de 9×7 cases autour du joueur et,
  en dessous, l'inventaire sur 9×2 cases : vie, nourriture, eau, énergie, objets.
- **17 actions** : ne rien faire ; se déplacer (4 directions) ; `do`, qui frappe, récolte, boit ou
  mange selon la case en face ; dormir ; poser une pierre, un établi, un four ou une pousse ;
  fabriquer une pioche ou une épée en bois, en pierre ou en fer.
- **Survie** : nourriture, eau et énergie baissent avec le temps. Si l'une tombe à 0, la vie
  baisse ; si elles sont toutes au-dessus de 0, la vie remonte. Les zombies (sur l'herbe) et les
  squelettes (dans les grottes) attaquent ; marcher dans la lave tue sur le coup.
- **Fin de partie** : à la mort, ou après 10 000 pas.
- **Récompense** :
  - +1 la première fois qu'un des 22 succès est obtenu dans la partie ;
  - ±0,1 par point de vie gagné ou perdu.

  Le reward maximum d'une partie vaut donc environ 22.
- **Les 22 succès**, qui forment un arbre technologique :
  - collecter : bois, pousse, eau, pierre, charbon, fer, diamant ;
  - placer : établi, pierre, pousse, four ;
  - fabriquer : pioche et épée en bois, puis en pierre, puis en fer ;
  - manger une vache, manger une plante ;
  - tuer un zombie, tuer un squelette ;
  - se réveiller après avoir dormi.

  Exemple de chaîne : bois → établi → pioche en bois → pierre → pioche en pierre → fer (il faut
  aussi un four et du charbon) → pioche en fer → diamant.
- **Score Crafter** : moyenne géométrique des taux de réussite des 22 succès, décalée de 1 %
  (`exp(moyenne(ln(1 + s_i))) − 1`). Elle est calculée sur tous les épisodes d'entraînement du
  premier million de pas. Avec la moyenne géométrique, débloquer un succès rare compte beaucoup
  plus que réussir encore mieux un succès facile. C'est la métrique du benchmark, plus
  informative que le reward.

## 2. Vue d'ensemble du pipeline

```
train_ppo.py / train_dqn.py
   │
   ├── Env("train")  ─ src/crafter_wrapper.py
   │      crafter.Env()                → image RGB 64×64, uint8
   │      crafter.Recorder(...)        → écrit logdir/.../stats.jsonl (1 ligne par épisode)
   │      GrayScale                    → moyenne des 3 canaux → 64×64
   │      Env (pile de 4 images, /255) → tenseur float32 (4, 64, 64), entre 0 et 1
   │
   ├── Env("eval")  ─ même chose, sans Recorder
   │
   ├── agent  ─ agents/*.py + networks/*.py (+ memories/ppo_memory.py)
   │
   └── boucle : agir → env.step → mémoriser → apprendre → évaluer tous les --eval-interval pas
```

- **Un seul environnement**, joué pas à pas. L'agent choisit chaque action avec un passage du
  réseau sur une seule image (batch de 1).
- **Évaluation** : tous les `--eval-interval` pas, `eval()` joue `--eval-episodes` parties (20
  par défaut) sur l'environnement d'évaluation et ajoute le reward moyen à
  `logdir/.../eval_stats.pkl`.
- **Empilement de 4 images** : il donne une notion de mouvement, par exemple pour voir dans quelle
  direction va un zombie. Au début d'une partie, les 3 images précédentes sont des zéros.
- **Niveaux de gris** : le passage en gris réduit l'entrée par 3. En contrepartie, il perd la
  couleur, qui distingue des éléments importants : le fer (orangé) du charbon (noir) et de la
  pierre (gris), l'eau de l'herbe la nuit, etc.

## 3. Le réseau convolutif (commun aux trois agents)

Architecture « Nature DQN » (Mnih et al., 2015) :

| Couche | Paramètres | Sortie |
|---|---|---|
| Entrée | 4 images 64×64 | 4 × 64 × 64 |
| Conv 1 | 32 filtres 8×8, pas 4, ReLU | 32 × 15 × 15 |
| Conv 2 | 64 filtres 4×4, pas 2, ReLU | 64 × 6 × 6 |
| Conv 3 | 64 filtres 3×3, pas 1, ReLU | 64 × 4 × 4 = 1 024 |
| Dense | 1 024 → 512, ReLU | 512 |
| Sortie | 512 → 17 Q-values (DQN), 17 probabilités (acteur PPO) ou 1 valeur (critique PPO) | |

Cela fait environ 0,6 million de paramètres par réseau, dont la majorité dans la couche dense
(1 024 × 512). Une remarque utile pour la suite : avec un pas de 4 dès la première couche, chaque
case du jeu (environ 7×7 pixels) est fortement sous-échantillonnée. Le réseau voit le monde de
façon très grossière.

## 4. DQN — `agents/dqn_agent.py`, lancé par `train_dqn.py --agent dqn`

**Principe.** On apprend Q(s, a), la somme des récompenses futures (actualisées par γ) si l'on
joue a dans l'état s puis qu'on continue au mieux. La cible vient de l'équation de Bellman :
`Q(s,a) ← r + γ · max_a' Q(s', a')`.

**Implémentation.**
- **Mémoire de replay** : des tableaux numpy de 100 000 transitions (s, a, r, s', done). Les
  états s et s' sont stockés séparément en float32, ce qui fait environ 13 Go.
- **Choix de l'action** : ε-greedy. Avec une probabilité ε l'action est aléatoire, sinon c'est
  `argmax Q`.
- **Apprentissage** : à chaque pas d'environnement, dès que la mémoire contient 32 transitions,
  on fait une mise à jour sur un lot aléatoire de 32.
- **Cible** : `r + 0,99 · max Q(s') · (1 − done)`, calculée avec **le même réseau** que celui
  qu'on entraîne. Il n'y a pas de réseau cible.
- **Perte et optimiseur** : Huber (`SmoothL1Loss`) et RMSprop, avec un taux d'apprentissage de
  2,5e-4.
- **Exploration** : ε part de 1 et perd 5e-4 à chaque mise à jour, jusqu'à 0,1. Il atteint donc
  0,1 en environ 1 800 pas. La constante `EPSILON_DECAY = 1e-6` de `train_dqn.py` n'est jamais
  transmise à l'agent.
- **Évaluation** : la docstring parle d'une politique gloutonne, mais `act()` applique quand même
  ε. On évalue donc avec 10 % d'actions aléatoires.

**Ce qui pose problème.**
- **Sans réseau cible**, la cible bouge à chaque mise à jour : on court après sa propre queue.
  C'est l'instabilité classique que le DQN de DeepMind résout en figeant une copie du réseau
  pendant plusieurs milliers de pas.
- **L'exploration s'arrête quasiment tout de suite**, alors que Crafter demande beaucoup
  d'exploration.

## 5. « Double DQN » — `agents/ddqn_agent.py`, lancé par `train_dqn.py --agent ddqn`

**Principe attendu.** Double DQN (van Hasselt, 2016) corrige la surestimation de DQN. Le réseau
en ligne choisit la meilleure action suivante, et le réseau cible l'évalue :
`r + γ · Q_cible(s', argmax_a Q_en_ligne(s', a))`.

**Implémentation.**
- **Deux réseaux** : Q1, entraîné, et Q2, dont les gradients sont désactivés.
- **Copie** : après **chaque** mise à jour, Q2 reçoit une copie de Q1. Les deux réseaux sont donc
  toujours identiques.
- **Cible** : `r + γ · max Q2(s') · done`. C'est le `max` de DQN, pas la sélection et
  l'évaluation de Double DQN.
- **Perte** : erreur quadratique écrite à la main. RMSprop, même ε et même mémoire que DQN.

**Ce qui pose problème.**
1. **La cible est inversée** : `· done` au lieu de `· (1 − done)`. La valeur future n'est ajoutée
   que sur la transition qui termine l'épisode, c'est-à-dire précisément là où elle devrait valoir
   0. Partout ailleurs, la cible vaut juste r. L'agent apprend donc surtout la récompense
   immédiate, sans voir plus loin que le pas suivant.
2. **Pas de vrai réseau cible**, puisque Q2 = Q1 en permanence.
3. **Ce n'est pas l'algorithme Double DQN** : il n'y a pas de découplage entre le choix et
   l'évaluation de l'action.

Malgré cela, DDQN fait un peu mieux que DQN en 2022 (2,7 % contre 2,3 %). C'est cohérent : un
agent myope qui récolte ce qui rapporte tout de suite (pousse, bois, eau) bat un agent instable.

## 6. PPO — `agents/ppo_agent.py`, lancé par `train_ppo.py`

**Principe.** PPO (Schulman, 2017) apprend directement une politique π(a|s), appelée l'acteur,
avec une fonction de valeur V(s), appelée le critique. On joue un « rollout » avec la politique
actuelle. On calcule pour chaque pas l'avantage A, qui mesure de combien l'action a fait mieux
que prévu. On augmente ensuite la probabilité des actions à avantage positif, sans trop
s'éloigner de l'ancienne politique : c'est le rôle du *clip* du ratio π_nouveau / π_ancien
dans [0,8 ; 1,2].

**Implémentation.**
- **Deux réseaux séparés**, sans tronc commun : l'acteur (CNN puis softmax sur 17 actions) et le
  critique (CNN puis une valeur). Chacun a son optimiseur Adam (lr 3e-4).
- **Rollout** : 2 048 pas dans un seul environnement, puis apprentissage.
- **Avantages** : GAE avec γ = 0,99 et λ = 0,95. Ils sont calculés par une double boucle Python
  en O(n²), soit environ 2 millions d'itérations, **refaite à chacune des 10 epochs**, alors que
  le résultat est le même à chaque fois.
- **Mise à jour** : 10 epochs × 32 mini-lots de 64, soit 320 pas de gradient par rollout.
- **Perte** : `perte_acteur (clip 0,2) + 0,5 × perte_critique (MSE)`. Il n'y a **pas de bonus
  d'entropie** (le commentaire `# Entropy ?` est resté).
- **Gradients** : écrêtés à une norme de 0,5 pour chacun des deux réseaux.
- **Actions** : toujours tirées au hasard selon π, y compris en évaluation.
- **Sauvegarde** : à la toute fin seulement. En 2022 dans `checkpoints/ppo/` ; désormais dans `logdir/.../checkpoints/`, pour que des runs en parallèle ne s'écrasent pas.

**Ce qui pose problème.**
- **Sans bonus d'entropie**, la politique devient vite presque déterministe et cesse d'explorer.
- **Avantages non normalisés** : leur échelle varie d'un rollout à l'autre, ce qui rend le pas
  d'apprentissage effectif irrégulier.
- **Pas de bootstrap en fin de rollout** : l'avantage du dernier pas vaut 0 et les derniers pas
  sont sous-estimés. L'effet est faible avec 2 048 pas.
- **Un seul environnement** : les 2 048 pas d'un rollout sont très corrélés (une ou deux
  parties), ce qui donne peu de diversité par mise à jour.
- **Le GAE en O(n²)** est le principal coût de calcul de l'apprentissage, et il tourne sur CPU
  en Python pur.

PPO était votre meilleur agent (3,2 %). C'est aussi l'algorithme qui offre le plus de marge de
progression ici : le PPO publié avec Crafter fait 4,6 %, et un PPO bien réglé atteint environ
15 % au même budget.

## 7. Fichiers produits et outils d'analyse

| Fichier | Écrit par | Contenu |
|---|---|---|
| `logdir/<agent>/<n>/stats.jsonl` | `crafter.Recorder` | 1 ligne par épisode d'entraînement : durée, reward, compteur de chaque succès |
| `.../eval_stats.pkl` | `eval()` | reward moyen de chaque évaluation (pickles empilés) |
| `.../loss_log.txt`, `q_values_log.txt` | DQN/DDQN | une valeur par mise à jour (fichiers énormes : environ 1M de lignes) |
| `.../actor_/critic_/total_losses_log.txt` | PPO | une valeur par rollout |
| `.../checkpoints/actor_torch_ppo`, `critic_torch_ppo` | PPO | poids des réseaux en fin d'entraînement |

- `analysis/crafter_score.py` : score Crafter officiel et taux par succès, à partir des
  `stats.jsonl`. Les scores du README en viennent.
- `analysis/plot_eval_performance.py` : courbe du reward d'évaluation (moyenne sur les runs),
  histogramme des succès, courbe de loss.

Pendant l'entraînement, le terminal affiche :
- tous les `--log-every` pas (5 000 par défaut), une ligne de progression : vitesse, temps
  restant estimé, reward moyen des 20 dernières parties, répartition du temps (voir section 9) ;
- tous les 20 épisodes, la ligne d'origine `Train mean_reward`, désactivable avec
  `--log-episodes 0` ;
- à chaque évaluation, la ligne `eval results: R/ep=…`.

## 8. Résultats de 2022

Score Crafter calculé sur le premier million de pas :

| Agent | Score | Profil (taux de réussite) |
|---|---|---|
| Aléatoire | 1,6 % | se réveiller 94 %, pousse 51 %, bois 24 %, eau 10 %, établi 4 % |
| DQN | 2,2 – 2,3 % | pousse 93 %, bois 30 %, eau 11 %, vache 8 %, établi 7 % |
| DDQN | 2,6 – 2,7 % | pousse 99 %, bois 34 %, établi 12 %, vache 11 %, zombie 10 % |
| PPO (meilleur run) | 3,2 % | se réveiller 94 %, bois 59 %, établi 31 %, eau 28 %, pioche en bois 5 % |

Aucun agent n'atteint de façon fiable la pierre : même le meilleur PPO laisse 12 des 22 succès
sous 1 %, dont 8 jamais obtenus.

## 9. Où passe le temps, et que peut apporter le GPU

Ordres de grandeur, à confirmer avec la ligne de progression sur votre machine :

- **`env`** : la simulation et le rendu de l'image par Crafter, en Python, sur un seul cœur CPU.
  Le GPU n'y change rien.
- **`act`** : un passage du CNN pour une seule image. Sur GPU, ce calcul minuscule est dominé par
  le coût fixe de chaque appel (copie vers le GPU, lancement des noyaux, retour avec `.item()`).
  Le gain est donc faible, voire nul.
- **`learn`** :
  - PPO : le GAE en O(n²) en Python domine et reste sur CPU. Seule la partie réseau (320 mini-lots
    de 64) profite du GPU.
  - DQN : une mise à jour par pas, sur 32 images. Le GPU accélère le réseau, mais il faut d'abord
    copier les lots depuis la mémoire numpy (environ 4 Mo par pas).

Conclusion attendue : à l'état 0, le GPU apporte un gain modeste. Il ne deviendra décisif que
lorsqu'on fera tourner plusieurs environnements en parallèle avec des lots d'images plus gros, et
que les calculs Python comme le GAE seront vectorisés. Le README explique comment le mesurer.
