import sys
import math
import random
import os
import numpy as np
import pygame
from numba import njit

# ── Constantes ──────────────────────────────────────────────────────────────

WALKABLE_INIT   = 100
BLOCKED_INIT    = 999
TRUCK_SPEED     = 4.0        
LOAD_TIME       = 1.0        
UNLOAD_TIME     = 0.8        
WOOD_PER_FOREST = 2          
MAX_TRUCKS      = 5          
SPAWN_INTERVAL  = 5.0        
DEFAULT_ZOOM    = 45         
MIN_ZOOM        = 20

# Dossier des ressources (images)
ASSETS_DIR      = "assets"


# ── Carte de Distance avec Numba (Algorithme Wavefront) ──────────────────────

@njit(cache=True)
def compute_distance_map_numba(base_dist, targets_xy, max_iterations, out_dist):
    """
    Calcule la carte des distances à partir des cibles en utilisant l'algorithme Wavefront.
    Optimisé avec Numba pour des performances maximales.
    """
    w, h = base_dist.shape
    for x in range(w):
        for y in range(h):
            out_dist[x, y] = base_dist[x, y]

    n_targets = targets_xy.shape[0]
    for i in range(n_targets):
        tx = targets_xy[i, 0]
        ty = targets_xy[i, 1]
        if 0 <= tx < w and 0 <= ty < h:
            out_dist[tx, ty] = 0

    changed = True
    iterations = 0
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for x in range(w):
            for y in range(h):
                if base_dist[x, y] != WALKABLE_INIT:
                    continue
                mn = BLOCKED_INIT
                if x + 1 < w  and out_dist[x+1, y] < mn: mn = out_dist[x+1, y]
                if x - 1 >= 0 and out_dist[x-1, y] < mn: mn = out_dist[x-1, y]
                if y + 1 < h  and out_dist[x, y+1] < mn: mn = out_dist[x, y+1]
                if y - 1 >= 0 and out_dist[x, y-1] < mn: mn = out_dist[x, y-1]
                new_val = mn + 1
                if new_val < out_dist[x, y]:
                    out_dist[x, y] = new_val
                    changed = True
    return iterations


# ── Structure de la Carte (20x12) ───────────────────────────────────────────
# Des emplacements pour les scieries (S) et les fabriques (M) ont été ajoutés.

# MAP_TEXT = """
# ####################
# #FFFFFF#           #
# #FFFFFF# ##### ### #
# ###### # #   # # # #
# #      # # DD# # # #
# # SSS      DD# MMM #
# # SSS  #   ### MMM #
# # #  ######### ### #
# # #              # #
# #     ########## # #
# #######       #VVVV#
# ####################
# """
MAP_TEXT = """
####################
#FFFFFF#           #
#FFFFFF# ##### ### #
###### # #   # # # #
#      # # DD# # # #
# SSS      DD# MMM #
# SSS  #   ### MMM #
# #  ######### ### #
# #              # #
#     ########## # #
#######       #VVVV#
####################
"""
class GameData:
    def __init__(self, map_text):
        self.map, self.mapW, self.mapH = self._parse(map_text)
        self.base_dist = self._build_base_dist()

        # Initialisation des forêts (F)
        self.forests = {}
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'F':
                    self.forests[(x, y)] = WOOD_PER_FOREST

        # Initialisation des scieries (S)
        self.scieries = []
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'S':
                    self.scieries.append((x, y))

        # Initialisation des fabriques de meubles (M)
        self.factories = []
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'M':
                    self.factories.append((x, y))

        # Initialisation des villes (V)
        self.cities = {}
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'V':
                    self.cities[(x, y)] = 0

        # Initialisation des dépôts (D) - Zone de spawn
        self.depots = []
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'D':
                    self.depots.append((x, y))

        self.total_furniture_delivered = 0

    def _parse(self, text):
        rows = [line for line in text.strip().splitlines() if line.strip()]
        maxlen = max(len(r) for r in rows)
        
        parsed = []
        for row in rows:
            cells = []
            for c in row.ljust(maxlen):
                if c in ('#', 'F', 'V', 'D', 'S', 'M'):
                    cells.append(c)
                else:
                    cells.append(' ')
            parsed.append(cells)

        arr = np.array(parsed, dtype='U1').T
        w, h = arr.shape
        return arr, w, h

    def _build_base_dist(self):
        base = np.full((self.mapW, self.mapH), BLOCKED_INIT, dtype=np.int32)
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] in (' ', 'D', 'F', 'V', 'S', 'M'):
                    base[x, y] = WALKABLE_INIT
        return base

    def computeDistanceMap(self, targets_xy, out_dist):
        if targets_xy is None or targets_xy.shape[0] == 0:
            out_dist[:, :] = self.base_dist
            return
        max_it = self.mapW * self.mapH
        compute_distance_map_numba(self.base_dist, targets_xy, max_it, out_dist)

    def available_forests(self):
        return [(pos, wood) for pos, wood in self.forests.items() if wood > 0]

    def available_scieries(self):
        return self.scieries

    def available_factories(self):
        return self.factories

    def available_cities(self):
        return list(self.cities.keys())

    def harvest(self, pos):
        if self.forests.get(pos, 0) > 0:
            self.forests[pos] -= 1
            return True
        return False

    def deliver(self, pos):
        if pos in self.cities:
            self.cities[pos] += 1
            self.total_furniture_delivered += 1

    def forest_exhausted(self, pos):
        return self.forests.get(pos, 0) <= 0


# ── Gestion de l'Écran et des Graphismes ─────────────────────────────────────

ZOOM = DEFAULT_ZOOM

class Screen:
    def __init__(self, nx, ny):
        pygame.display.init()
        pygame.font.init()
        global ZOOM
        info = pygame.display.Info()
        max_w = max(320, info.current_w - 60)
        max_h = max(240, info.current_h - 120)
        ZOOM = max(MIN_ZOOM, min(DEFAULT_ZOOM, max_w // nx, max_h // (ny + 1)))

        self.W = nx * ZOOM
        self.H = (ny + 1) * ZOOM 
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Transport Tycoon - Chaîne Logistique")
        self.clock = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Arial", max(12, int(ZOOM * 0.45)), bold=True)
        self.font_s = pygame.font.SysFont("Arial", max(10, int(ZOOM * 0.35)))

        # Chargement et redimensionnement des images de camions
        self.truck_images = []
        image_names = ["car-truck2.png", "car-truck3.png", "car-truck4.png", "car-truck5.png"]
        
        for name in image_names:
            path = os.path.join(ASSETS_DIR, name)
            try:
                img = pygame.image.load(path).convert_alpha()
                
                orig_w, orig_h = img.get_size()
                target_w = int(ZOOM * 0.55) 
                target_h = int(target_w * (orig_h / orig_w))
                
                if target_h > int(ZOOM * 0.85):
                    target_h = int(ZOOM * 0.85)
                    target_w = int(target_h * (orig_w / orig_h))

                img = pygame.transform.scale(img, (target_w, target_h))
                self.truck_images.append(img)
            except Exception as e:
                print(f"Erreur : Impossible de charger {path} ! Échec. Rapport : {e}")
                fallback = pygame.Surface((int(ZOOM * 0.5), int(ZOOM * 0.8)), pygame.SRCALPHA)
                pygame.draw.rect(fallback, (200, 50, 50), (0, 0, int(ZOOM * 0.5), int(ZOOM * 0.8)), border_radius=3)
                self.truck_images.append(fallback)

        # ── CHARGEMENT DE BRICKHOUSE.PNG (Villes) ──
        self.city_asset = None
        city_asset_path = os.path.join(ASSETS_DIR, "BrickHouse.png")
        try:
            self.city_asset = pygame.image.load(city_asset_path).convert_alpha()
            self.city_asset = pygame.transform.scale(self.city_asset, (int(ZOOM * 0.8), int(ZOOM * 0.8)))
        except Exception as e:
            print(f"Erreur : Impossible de charger BrickHouse.png ! {e}")

        # ── CHARGEMENT DE FACTORY.PNG & ISOMETRIC_OFFICE_5.PNG ──
        # ── CHARGEMENT DE FACTORY.PNG ──
        self.factory_asset = None
        self.office_asset = None
        try:
            self.factory_asset = pygame.image.load(os.path.join(ASSETS_DIR, "factory.png")).convert_alpha()
            # On redimensionne pour couvrir 6 cases (3x2). On donne un peu plus de hauteur (2.5) pour l'effet de relief.
            self.factory_asset = pygame.transform.scale(self.factory_asset, (int(ZOOM * 3), int(ZOOM * 2.5)))
        except Exception as e:
            print(f"Erreur : Impossible de charger factory.png : {e}")
            
        # ── CHARGEMENT DE ISOMETRIC_OFFICE_5.PNG ──
        try:
            self.office_asset = pygame.image.load(os.path.join(ASSETS_DIR, "isometric_office_5.png")).convert_alpha()
            # On redimensionne pour couvrir 6 cases (3x2).
            self.office_asset = pygame.transform.scale(self.office_asset, (int(ZOOM * 3), int(ZOOM * 2.5)))
        except Exception as e:
            print(f"Erreur : Impossible de charger isometric_office_5.png : {e}")

        # ── CHARGEMENT DES IMAGES D'ARBRES ──
        self.pintree_img = None
        self.deadtree_img = None
        try:
            pintree = pygame.image.load(os.path.join(ASSETS_DIR, 'pintree.png')).convert_alpha()
            deadtree = pygame.image.load(os.path.join(ASSETS_DIR, 'deadtree.png')).convert_alpha()
            self.pintree_img = pygame.transform.scale(pintree, (ZOOM, ZOOM))
            self.deadtree_img = pygame.transform.scale(deadtree, (ZOOM, ZOOM))
        except Exception as e:
            print(f"Info : Impossible de charger les images d'arbres ({e}).")

    def grid_to_screen(self, x, y):
        return int(x * ZOOM), int(y * ZOOM)

    def drawRect(self, x, y, color=(0,0,0), border=0):
        sx, sy = self.grid_to_screen(x, y)
        pygame.draw.rect(self.screen, color, (sx, sy, ZOOM, ZOOM), border)

    def drawText(self, x, y, txt, color=(255,255,255), big=True, centered=False):
        font = self.font if big else self.font_s
        sx, sy = self.grid_to_screen(x, y)
        surf = font.render(str(txt), True, color)
        if centered:
            ox = ZOOM // 2 - surf.get_width() // 2
            oy = ZOOM // 2 - surf.get_height() // 2
            self.screen.blit(surf, (sx + ox, sy + oy))
        else:
            self.screen.blit(surf, (sx, sy))

    def drawImage(self, x, y, image):
        sx, sy = self.grid_to_screen(x, y)
        rect = image.get_rect(center=(sx + ZOOM // 2, sy + ZOOM // 2))
        self.screen.blit(image, rect.topleft)

    def drawTruckSprite(self, x, y, dx, dy, img_index, cargo_type):
        cx, cy = int(x * ZOOM), int(y * ZOOM)
        base_img = self.truck_images[img_index % len(self.truck_images)]

        angle = 0
        if dy < 0:    angle = 0    
        elif dy > 0:  angle = 180  
        elif dx > 0:  angle = 270  
        elif dx < 0:  angle = 90   

        rotated_img = pygame.transform.rotate(base_img, angle)
        rect = rotated_img.get_rect(center=(cx, cy))
        self.screen.blit(rotated_img, rect.topleft)

        # Dessin d'un point de couleur sur le camion selon le type de cargaison
        if cargo_type == "wood":
            pygame.draw.circle(self.screen, (139, 90, 43), (cx, cy), 4) # Bois (Marron)
        elif cargo_type == "planks":
            pygame.draw.circle(self.screen, (222, 184, 135), (cx, cy), 4) # Planches (Jaunâtre)
        elif cargo_type == "furniture":
            pygame.draw.circle(self.screen, (255, 140, 0), (cx, cy), 4) # Meubles (Orange)

    def drawCitySprite(self, x, y):
        if self.city_asset:
            sx, sy = self.grid_to_screen(x, y)
            rect = self.city_asset.get_rect(center=(sx + ZOOM // 2, sy + ZOOM // 2))
            self.screen.blit(self.city_asset, rect.topleft)

    def drawScierieSprite(self, x, y):
        if self.factory_asset:
            sx, sy = self.grid_to_screen(x, y)
            rect = self.factory_asset.get_rect(center=(sx + ZOOM // 2, sy + ZOOM // 2))
            self.screen.blit(self.factory_asset, rect.topleft)

    def drawLargeScierieSprite(self, x, y):
        if self.factory_asset:
            # x et y correspondent au coin supérieur gauche du bloc de 6 cases (S)
            sx, sy = self.grid_to_screen(x, y)
            
            # On aligne le bas de l'image avec le bas du bloc de 2 cases de hauteur (ZOOM * 2)
            rect = self.factory_asset.get_rect()
            rect.bottomleft = (sx, sy + (ZOOM * 2))
            self.screen.blit(self.factory_asset, rect.topleft)

    def drawFactorySprite(self, x, y):
        if self.office_asset:
            sx, sy = self.grid_to_screen(x, y)
            rect = self.office_asset.get_rect(center=(sx + ZOOM // 2, sy + ZOOM // 2))
            self.screen.blit(self.office_asset, rect.topleft)

    def drawLargeFactorySprite(self, x, y):
            if self.office_asset:
                # x et y correspondent exactement au coin supérieur gauche du bloc M
                sx, sy = self.grid_to_screen(x, y)
                
                # On réinitialise le rect pour éviter toute superposition ou décalage
                rect = self.office_asset.get_rect()
                # On cale le coin supérieur gauche de l'image sur le coin supérieur gauche du grid
                # On applique un léger offset vertical (- int(ZOOM * 0.5)) uniquement pour l'effet de toit izometrik
                rect.topleft = (sx, sy - int(ZOOM * 0.5))
                
                self.screen.blit(self.office_asset, rect.topleft)

    def show(self):
        pygame.display.flip()


# ── Classe Camion (Truck) ───────────────────────────────────────────────────

class Truck:
    _id_counter = 0

    def __init__(self, game: GameData, depot_pos):
        Truck._id_counter += 1
        self.tid  = Truck._id_counter
        self.img_index = (self.tid - 1) % 4

        self.cx, self.cy = depot_pos     
        self.nx, self.ny = depot_pos     
        self.x = self.cx + 0.5
        self.y = self.cy + 0.5
        
        self.dir  = (0.0, -1.0) 
        self.speed = TRUCK_SPEED + random.uniform(-0.2, 0.2)

        self.cargo = None             # None, "wood", "planks", "furniture"
        self.state = "seeking_forest" # État initial
        self.timer = 0.0
        self.target_pos = None

        self.dist = np.empty_like(game.base_dist)
        self._pick_forest_target(game)

    def _pick_forest_target(self, game):
        avail = game.available_forests()
        if not avail:
            self.state = "idle"
            self.target_pos = None
            return

        best, best_score = None, -1
        for (fx, fy), wood in avail:
            dist = abs(fx - self.cx) + abs(fy - self.cy) or 1
            score = wood / dist
            if score > best_score:
                best_score = score
                best = (fx, fy)

        self.target_pos = best
        self.state = "seeking_forest"
        self._refresh_dist(game)

    def _pick_scierie_target(self, game):
        scieries = game.available_scieries()
        if not scieries:
            self.state = "idle"
            self.target_pos = None
            return
        self.target_pos = min(scieries, key=lambda p: abs(p[0]-self.cx) + abs(p[1]-self.cy))
        self.state = "seeking_scierie"
        self._refresh_dist(game)

    def _pick_factory_target(self, game):
        factories = game.available_factories()
        if not factories:
            self.state = "idle"
            self.target_pos = None
            return
        self.target_pos = min(factories, key=lambda p: abs(p[0]-self.cx) + abs(p[1]-self.cy))
        self.state = "seeking_factory"
        self._refresh_dist(game)

    def _pick_city_target(self, game):
        cities = game.available_cities()
        if not cities:
            self.state = "idle"
            self.target_pos = None
            return

        best = min(cities, key=lambda p: abs(p[0]-self.cx) + abs(p[1]-self.cy))
        self.target_pos = best
        self.state = "seeking_city"
        self._refresh_dist(game)

    def _refresh_dist(self, game):
        if self.target_pos is None:
            game.computeDistanceMap(np.empty((0, 2), dtype=np.int32), self.dist)
            return
        tx, ty = self.target_pos
        targets = np.array([[tx, ty]], dtype=np.int32)
        game.computeDistanceMap(targets, self.dist)

    def _is_walkable(self, game, x, y):
        if not (0 <= x < game.mapW and 0 <= y < game.mapH):
            return False
        return game.map[x, y] != '#'

    def _next_grid_step(self, game):
        best_d = BLOCKED_INIT
        cands = []
        
        for ddx, ddy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = self.cx + ddx, self.cy + ddy
            if not self._is_walkable(game, nx, ny):
                continue
            nd = self.dist[nx, ny]
            if nd < best_d:
                best_d = nd
                cands = [(nx, ny)]
            elif nd == best_d:
                cands.append((nx, ny))
                
        if cands and best_d < BLOCKED_INIT:
            self.nx, self.ny = random.choice(cands)
        else:
            self.nx, self.ny = self.cx, self.cy

    def update(self, game: GameData, dt):
        if self.state == "idle":
            if game.available_forests():
                self._pick_forest_target(game)
            return

        if self.state == "loading":
            self.timer -= dt
            if self.timer <= 0:
                if self.cargo is None:
                    self.cargo = "wood"
                    self._pick_scierie_target(game)
                elif self.cargo == "wood":
                    self.cargo = "planks"
                    self._pick_factory_target(game)
                elif self.cargo == "planks":
                    self.cargo = "furniture"
                    self._pick_city_target(game)
                return

        if self.state == "unloading":
            self.timer -= dt
            if self.timer <= 0:
                if self.cargo == "wood":
                    self.state = "loading"
                    self.timer = LOAD_TIME
                elif self.cargo == "planks":
                    self.state = "loading"
                    self.timer = LOAD_TIME
                elif self.cargo == "furniture":
                    game.deliver(self.target_pos)
                    self.cargo = None
                    self._pick_forest_target(game)
                return

        if self.cx == self.nx and self.cy == self.ny:
            if self.target_pos and (abs(self.target_pos[0] - self.cx) + abs(self.target_pos[1] - self.cy) <= 1):
                if self.state == "seeking_forest":
                    fp = self.target_pos
                    if fp and not game.forest_exhausted(fp):
                        game.harvest(fp)
                        self.state = "loading"
                        self.timer = LOAD_TIME
                    else:
                        self._pick_forest_target(game)
                elif self.state == "seeking_scierie":
                    self.state = "unloading"
                    self.timer = UNLOAD_TIME
                elif self.state == "seeking_factory":
                    self.state = "unloading"
                    self.timer = UNLOAD_TIME
                elif self.state == "seeking_city":
                    self.state = "unloading"
                    self.timer = UNLOAD_TIME
                return
            
            self._next_grid_step(game)

        target_x = self.nx + 0.5
        target_y = self.ny + 0.5
        
        vx = target_x - self.x
        vy = target_y - self.y
        dist = math.hypot(vx, vy)

        step = self.speed * dt
        if step >= dist:
            self.x = target_x
            self.y = target_y
            self.cx = self.nx
            self.cy = self.ny
        else:
            if dist > 0:
                self.dir = (vx / dist, vy / dist)
            self.x += self.dir[0] * step
            self.y += self.dir[1] * step

    def draw(self, S: Screen):
        S.drawTruckSprite(self.x, self.y, self.dir[0], self.dir[1], self.img_index, self.cargo)


# ── Éléments de Dessin et d'Arrière-Plan ─────────────────────────────────────

COLOR_ROAD         = (235, 230, 225)
COLOR_WALL         = (110, 110, 110)   
COLOR_FOREST_EMPTY = (150, 125, 100)
COLOR_CITY_BASE    = (205, 195, 175)
COLOR_CITY_DONE    = (100, 210, 130)
COLOR_DEPOT        = (140, 140, 185)   
COLOR_GRID         = (220, 215, 205)

def build_background(game: GameData, S: Screen):
    surf = pygame.Surface((S.W, S.H))
    surf.fill((30, 30, 30))

    for x in range(game.mapW):
        for y in range(game.mapH):
            c = game.map[x, y]
            if c == '#':
                color = COLOR_WALL
            elif c in ('F', 'S', 'M'):
                color = COLOR_ROAD
            elif c == 'V':
                color = COLOR_CITY_BASE
            elif c == 'D':
                color = COLOR_DEPOT
            else:
                color = COLOR_ROAD

            sx, sy = S.grid_to_screen(x, y)
            pygame.draw.rect(surf, color, (sx, sy, ZOOM, ZOOM))
            if c != '#':
                pygame.draw.rect(surf, COLOR_GRID, (sx, sy, ZOOM, ZOOM), 1)
    return surf

def draw_map(game, S, background, trucks, spawn_timer):
    S.screen.blit(background, (0, 0))

    # ── DESSIN DE LA GRANDE SCIERIE (S) - UNE SEULE ET UNIQUE IMAGE ──
    if game.scieries:
        # On trie pour choper le pixel le plus en haut à gauche
        sorted_s = sorted(game.scieries, key=lambda p: (p[0], p[1]))
        first_s = sorted_s[0]
        S.drawLargeScierieSprite(first_s[0], first_s[1])

    # ── DESSIN DE LA GRANDE FABRIQUE DE MEUBLES (M) - UNE SEULE ET UNIQUE IMAGE ──
    if game.factories:
        # On trie rigoureusement par axe X puis par axe Y pour choper le pixel le plus en haut à gauche
        factories_sorted = sorted(game.factories, key=lambda p: (p[0], p[1]))
        exact_top_left_m = factories_sorted[0]
        
        first_m_x = exact_top_left_m[0]
        first_m_y = exact_top_left_m[1]
        
        # ON APPELLE LA FONCTION UNE SEULE FOIS (DÖNGÜ KALDIRILDI)
        S.drawLargeFactorySprite(first_m_x, first_m_y)

    # ── IMPORTANT : Eski "for (x, y) in game.factories:" döngüsü üst üste binmeyi önlemek için tamamen silindi ──

    # Affichage des villes (V)
    for (x, y), count in game.cities.items():
        if count > 0:
            ratio = min(1.0, count / 5.0)
            color = (
                int(COLOR_CITY_BASE[0] + ratio*(COLOR_CITY_DONE[0]-COLOR_CITY_BASE[0])),
                int(COLOR_CITY_BASE[1] + ratio*(COLOR_CITY_DONE[1]-COLOR_CITY_BASE[1])),
                int(COLOR_CITY_BASE[2] + ratio*(COLOR_CITY_DONE[2]-COLOR_CITY_BASE[2]))
            )
            S.drawRect(x, y, color=color)
            S.drawRect(x, y, color=COLOR_GRID, border=1)
            
        S.drawCitySprite(x, y)

    # Rendu des symboles de forêt
    for (x, y), wood in game.forests.items():
        if S.pintree_img and S.deadtree_img:
            if wood == WOOD_PER_FOREST: # 2 interactions restantes
                S.drawImage(x, y, S.pintree_img)
            elif wood > 0:
                S.drawImage(x, y, S.deadtree_img)
            else:
                pass
        else:
            # Rendu textuel de secours (Fallback) si les images ne sont pas chargées
            if wood == WOOD_PER_FOREST:
                S.drawText(x, y, "🌲", big=False, centered=True)
            elif wood > 0:
                S.drawText(x, y, "🌱", big=False, centered=True)
            else:
                S.drawText(x, y, "❌", big=False, centered=True)

    # Dessiner les camions
    for truck in trucks:
        truck.draw(S)

    # Barre de statut inférieure
    bar_y = S.H - ZOOM
    pygame.draw.rect(S.screen, (25, 25, 25), (0, bar_y, S.W, ZOOM))
    total = game.total_furniture_delivered
    remaining = sum(game.forests.values())
    
    if len(trucks) < MAX_TRUCKS:
        timer_text = f" | Nouveau véhicule : {max(0.0, spawn_timer):.1f}s"
    else:
        timer_text = " | Garage Plein (Max 5)"

    txt = f"Livraisons : {total}  |  Bois Restant : {remaining}  |  Camions : {len(trucks)}/{MAX_TRUCKS}{timer_text}"
    surf = S.font.render(txt, True, (240, 240, 240))
    S.screen.blit(surf, (15, bar_y + (ZOOM // 2 - surf.get_height() // 2)))

    S.show()

# ── Boucle Principale (Main Loop) ───────────────────────────────────────────

def main():
    pygame.init()
    game = GameData(MAP_TEXT)
    S    = Screen(game.mapW, game.mapH)

    depots = game.depots if game.depots else [(game.mapW//2, game.mapH//2)]
    trucks = [Truck(game, depots[0])]
    spawn_timer = SPAWN_INTERVAL

    background = build_background(game, S)

    PAUSE = False
    LOGIC_FPS = 30
    LOGIC_TIMER = pygame.USEREVENT + 1
    pygame.time.set_timer(LOGIC_TIMER, int(1000 / LOGIC_FPS))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    PAUSE = not PAUSE
                elif event.key == pygame.K_r:
                    game = GameData(MAP_TEXT)
                    Truck._id_counter = 0
                    trucks = [Truck(game, depots[0])]
                    spawn_timer = SPAWN_INTERVAL
                    background = build_background(game, S)
            elif event.type == LOGIC_TIMER:
                if not PAUSE:
                    dt = 1.0 / LOGIC_FPS
                    
                    for truck in trucks:
                        truck.update(game, dt)
                    
                    if len(trucks) < MAX_TRUCKS:
                        spawn_timer -= dt
                        if spawn_timer <= 0:
                            dp = random.choice(depots)
                            trucks.append(Truck(game, dp))
                            spawn_timer = SPAWN_INTERVAL
                    
                    background = build_background(game, S)

        draw_map(game, S, background, trucks, spawn_timer)
        S.clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()