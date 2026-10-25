# Crafter — agents d'apprentissage par renforcement

Agents DQN, Double DQN et PPO entraînés sur [Crafter](https://github.com/danijar/crafter), un
Minecraft 2D conçu comme benchmark : l'agent doit survivre (manger, boire, dormir, se défendre) et
enchaîner 22 succès de difficulté croissante (bois → établi → pioche → pierre → fer → diamant).

Le projet est parti d'un devoir de master en IA (2022). L'objectif est maintenant de l'améliorer
**pas à pas**, en comprenant à chaque étape ce qui change et pourquoi.

## Point de départ (2022)

Score Crafter : moyenne géométrique des taux de réussite des 22 succès, sur tous les épisodes
d'entraînement du premier million de pas. Calculé par `analysis/crafter_score.py` sur les runs
de `results/2022/runs/`.

| Agent | Runs | Score Crafter | Reward moyen (100 derniers épisodes) |
|---|---|---|---|
| Aléatoire | 4 | 1,6 % | 1,4 |
| DQN | 2 (≤ 600k pas) | 2,2 – 2,3 % | 2,6 – 2,8 |
| Double DQN | 2 | 2,6 – 2,7 % | 2,7 – 3,1 |
| **PPO** | 4 | **2,6 – 3,2 %** | 2,4 – 3,2 |
| *Références publiées (pixels, 1M pas)* | | PPO 4,6 % · DreamerV3 14,8 % · humain 50,5 % | |

Le meilleur run PPO coupe du bois (59 % des parties) et pose un établi (31 %), mais ne fabrique
presque jamais d'outils (5 %) et n'atteint pas la pierre.

## Installation (Windows, conda)

Le code tourne nativement sous Windows : pas besoin de Linux ni de WSL pour l'instant.

```powershell
conda activate crafterEnv          # environnement existant ; sinon : conda create -n crafterEnv python=3.11
cd Crafter-game

# 1) PyTorch avec CUDA (build CUDA 13.0 : `nvidia-smi` doit afficher une version CUDA ≥ 13.0). Vérifiez la commande exacte sur
#    https://pytorch.org/get-started/locally/ (Windows / Pip / CUDA le plus récent).
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 2) Le reste
pip install -r requirements.txt

# 3) Vérifications
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import crafter; e = crafter.Env(); print(e.reset().shape)"
```

La première commande de vérification doit afficher `True` et `NVIDIA GeForce RTX 4090`. Si elle
affiche `False`, c'est la version CPU de PyTorch qui est installée : désinstallez-la
(`pip uninstall torch`) puis relancez l'étape 1.

## Lancer le code

Tests rapides (une à deux minutes chacun) pour vérifier que tout fonctionne :

```powershell
python train_ppo.py --steps 4096 --eval-interval 4096 --eval-episodes 2 --logdir logdir/smoke/ppo/0
python train_dqn.py --agent ddqn --steps 3000 --eval-interval 3000 --eval-episodes 2 --logdir logdir/smoke/ddqn/0
```

Entraînements complets (1M pas, comme le benchmark) :

```powershell
python train_ppo.py --steps 1000000 --eval-interval 50000 --logdir logdir/ppo_agent/1
python train_dqn.py --agent dqn  --steps 1000000 --eval-interval 50000 --logdir logdir/dqn_agent/1
python train_dqn.py --agent ddqn --steps 1000000 --eval-interval 50000 --logdir logdir/ddqn_agent/1

# plusieurs runs en parallèle (une fenêtre par run)
.\scripts\run_seeds.ps1 -Runs 3 -Steps 1000000 -EvalInterval 50000
```

Jouer soi-même (utile pour comprendre le jeu) : `pip install pygame-ce` une fois, puis `python -m crafter.run_gui`.

Résultats et graphiques :

```powershell
python analysis/crafter_score.py --logdir logdir --budget 1000000 --rates
python analysis/plot_eval_performance.py --logdir logdir/ppo_agent      # → analysis/ppo_agent/*.png
```

Chaque run écrit dans son dossier `logdir/<agent>/<n>/` :
- `stats.jsonl` : une ligne par épisode d'entraînement (durée, reward, succès) ;
- `eval_stats.pkl` : reward moyen des évaluations ;
- les losses.

PPO sauvegarde aussi ses réseaux dans `checkpoints/ppo/`.

**Mémoire** : DQN et DDQN réservent un replay buffer de 100 000 transitions × 2 × (4×64×64)
float32, soit environ **13 Go de RAM**. Réduisez `MAX_MEMORY` dans `train_dqn.py` si besoin.

## Organisation

```
train_ppo.py            boucle d'entraînement PPO        (ex-train.py)
train_dqn.py            boucle DQN / Double DQN (--agent) (ex-train_classic.py)
agents/                 dqn_agent, ddqn_agent, ppo_agent, random_agent
networks/               réseaux convolutifs (Q-network, acteur, critique)
memories/ppo_memory.py  buffer de rollout PPO
src/crafter_wrapper.py  wrapper Crafter : niveaux de gris, empilement de 4 images, logs
analysis/               crafter_score.py (score officiel), plot_eval_performance.py (courbes)
scripts/run_seeds.ps1   lancer plusieurs runs en parallèle
results/2022/           runs, checkpoints PPO et graphiques du devoir d'origine
docs/                   énoncé du devoir, règles, articles de référence
```

## Limites connues du code de 2022 (pistes de la prochaine étape)

Constatées à la relecture, **pas encore corrigées**, volontairement : le code actuel reste le
point de départ mesurable, et chaque correction sera une étape comparée à celui-ci.

- **Double DQN, cible inversée** (`agents/ddqn_agent.py`) : `q_target = r + γ·max Q'(s') · done`
  au lieu de `· (1 − done)`. La valeur future n'est comptée qu'en fin d'épisode, donc l'agent
  n'apprend pratiquement que la récompense immédiate.
- **Pas de vrai réseau cible** : en DDQN, Q2 est recopié sur Q1 à chaque pas ; DQN n'en a pas.
  L'apprentissage est donc instable.
- **Exploration DQN** : `EPSILON_DECAY` n'est pas transmis à l'agent. ε passe de 1 à 0,1 en
  environ 1 800 pas au lieu d'environ 900 000.
- **PPO** :
  - pas de bonus d'entropie, donc la politique se fige tôt ;
  - avantages non normalisés ;
  - GAE recalculé en O(n²) à chaque epoch (lent, même résultat qu'en O(n)) ;
  - pas de bootstrap de la valeur en fin de rollout ;
  - un seul environnement.
- **Observation en niveaux de gris** : la couleur distingue des éléments clés (fer, charbon,
  diamant, eau…), et le passage en gris la perd.
- **GPU sous-utilisé** : l'environnement tourne sur un seul cœur CPU, donc le GPU attend la
  plupart du temps. La parallélisation des environnements sera l'un des premiers gains.

L'agent M-DQN de 2022 a été retiré : il dépendait d'une classe `ReplayBuffer` absente du dépôt.
Il reste disponible dans l'historique git (premier commit).
