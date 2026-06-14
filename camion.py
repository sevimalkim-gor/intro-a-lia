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


MAP_TEXT = """
####################
#FFFFFF#       #WWW#
#FFFFFF# ##### #####
###    # #         #
#      # # DD# #####
# SSS      DD# MMM #
# SSS      ### MMM #
#     ####     ### #
# ##  #WW#       # #
#     ########## # #
#####         #VVVV#
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
        # ── CHARGEMENT DE BRICKTILES.PNG ──
        self.tile_img = None
        try:
            path = os.path.join(ASSETS_DIR, "BrickTiles.png")
            img = pygame.image.load(path).convert_alpha()
            
            # Eğer resim kare kare ayrılmış küçük bir tile ise, 
            # doğrudan her karenin içine tam oturması için ZOOM boyutuna pürüzsüz ölçekliyoruz.
            self.tile_img = pygame.transform.smoothscale(img, (ZOOM, ZOOM))
        except Exception as e:
            print(f"Erreur : Impossible de charger BrickTiles.png ! {e}")

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
        self.money = 0
        self.floating_texts = []

    def _parse(self, text):
        rows = [line for line in text.strip().splitlines() if line.strip()]
        maxlen = max(len(r) for r in rows)
        
        parsed = []
        for row in rows:
            cells = []
            for c in row.ljust(maxlen):
                if c in ('#', 'F', 'V', 'D', 'S', 'M', 'W'):
                    cells.append(c)
                else:
                    cells.append(' ')
            parsed.append(cells)

        arr = np.array(parsed, dtype='U1').T
        w, h = arr.shape
        return arr, w, h

    def _build_base_dist(self):
        # Varsayılan olarak tüm haritayı engelli (BLOCKED_INIT) yapıyoruz
        base = np.full((self.mapW, self.mapH), BLOCKED_INIT, dtype=np.int32)
        
        for x in range(self.mapW):
            for y in range(self.mapH):
                # S, M, # ve yeni eklediğimiz W harfi bu listede OLMADIĞI için 
                # otomatik olarak ENGELLİ (duvar gibi) kabul edilecekler.
                if self.map[x, y] in (' ', 'D', 'F', 'V'):
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
            
            self.money += 9
            self.floating_texts.append(FloatingText(pos[0], pos[1], "+$9"))

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
        self.font   = pygame.font.SysFont("Arial", max(12, int(ZOOM * 0.35)), bold=True)
        self.font_s = pygame.font.SysFont("Arial", max(10, int(ZOOM * 0.30)))

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

        # ── CHARGEMENT DE BRICKTILES.PNG ──

        self.tile_img = None
        try:
            path = os.path.join(ASSETS_DIR, "BrickTiles.png")
            img = pygame.image.load(path).convert_alpha()
            # Resmi tam olarak bir harita karesi (ZOOM x ZOOM) boyutuna getiriyoruz
            self.tile_img = pygame.transform.scale(img, (ZOOM, ZOOM))
        except Exception as e:
            print(f"Erreur : Impossible de charger BrickTiles.png ! {e}")

# ── 1. ÇİMENİ YÜKLE (Duvarlar için) ──
        self.grass_img = None
        try:
            path = os.path.join(ASSETS_DIR, "grass.png")
            img = pygame.image.load(path).convert_alpha()
            self.grass_img = pygame.transform.smoothscale(img, (ZOOM, ZOOM))
        except Exception as e:
            print(f"Grass yüklenemedi: {e}")

        # ── 2. SUYU YÜKLE (W kareleri için) ──
        # Burayı kontrol et, silinmiş veya ismi değişmiş olabilir!
        self.water_img = None
        try:
            path = os.path.join(ASSETS_DIR, "water.png")
            img = pygame.image.load(path).convert_alpha()
            self.water_img = pygame.transform.smoothscale(img, (ZOOM, ZOOM))
        except Exception as e:
            print(f"Water yüklenemedi: {e}")


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
    # ── CHARGEMENT DE TREASURE.PNG ──
        self.treasure_icon = None
        try:
            path = os.path.join(ASSETS_DIR, "treasure.png")
            img = pygame.image.load(path).convert_alpha()
            # Alt barın yüksekliğine (ZOOM değerine) göre resmi orantılı ölçekliyoruz (Yaklaşık %60'ı kadar)
            icon_size = int(ZOOM * 0.6)
            self.treasure_icon = pygame.transform.scale(img, (icon_size, icon_size))
        except Exception as e:
            print(f"Erreur : Impossible de charger treasure.png ! {e}")
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

# ── Classe Coin Text  ───────────────────────────────────────────────────

class FloatingText:
    def __init__(self, grid_x, grid_y, text, color=(255, 215, 0)):
        self.x = grid_x + 0.5  # Karenin ortasından başlasın
        self.y = grid_y
        self.text = text
        self.color = color
        self.lifetime = 1.0    # 1 saniye boyunca ekranda kalacak
        self.vel_y = -1.2      # Yukarı doğru yükselme hızı

    def update(self, dt):
        self.y += self.vel_y * dt
        self.lifetime -= dt

    def draw(self, S):
        # Yazının yukarı doğru çıkarken yavaşça şeffaflaşması (Fade-out efekti)
        alpha = max(0, min(255, int(self.lifetime * 255)))
        
        # Kalın ve biraz daha büyük fontla parayı basalım
        S.font.set_bold(True)
        surf = S.font.render(self.text, True, self.color)
        
        # Şeffaflık uygulamak için geçici bir yüzey kullanıyoruz
        alpha_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        alpha_surf.fill((255, 255, 255, alpha))
        surf.blit(alpha_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Piksel koordinatlarına çevirip ekrana basıyoruz
        sx = int(self.x * ZOOM) - surf.get_width() // 2
        sy = int(self.y * ZOOM)
        S.screen.blit(surf, (sx, sy))
        S.font.set_bold(False)

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
COLOR_WALL         = (35, 110, 55)   
COLOR_FOREST_EMPTY = (150, 125, 100)
COLOR_CITY_BASE    = (205, 195, 175)
COLOR_CITY_DONE    = (100, 210, 130)
COLOR_DEPOT        = (140, 140, 185)   
COLOR_GRID         = (220, 215, 205)

def build_background(game, S):
    bg = pygame.Surface((S.W, S.H))
    
    # --- 1. ADIM: TÜM HARİTAYI BRICKTILES (YOL) İLE DÖŞE ---
    if S.tile_img:
        for x in range(game.mapW):
            for y in range(game.mapH):
                bg.blit(S.tile_img, (x * ZOOM, y * ZOOM))
    else:
        bg.fill(COLOR_ROAD)

    # --- 2. ADIM: DUVARLARI (#) VE SULARI (W) DOKUYLA KAPLA ---
    for x in range(game.mapW):
        for y in range(game.mapH):
            char = game.map[x, y]
            px = x * ZOOM
            py = y * ZOOM

            # A) DUVARLAR (#) -> ARTIK ÇİMEN RESMİ
            if char == '#':
                if hasattr(S, 'grass_img') and S.grass_img:
                    bg.blit(S.grass_img, (px, py))
                else:
                    pygame.draw.rect(bg, COLOR_WALL, (px, py, ZOOM, ZOOM))
            

            # B) YENİ SU BÖLGELERİ (W)
            elif char == 'W':
                if hasattr(S, 'water_img') and S.water_img:
                    bg.blit(S.water_img, (px, py))
                else:
                    # EĞER RESMİ BULAMAZSA BURASI ÇALIŞACAK: 
                    # Haritada parlak kırmızı kareler görüyorsan bil ki sorun RESMİN YÜKLENEMEMESİDİR.
                    pygame.draw.rect(bg, (255, 0, 0), (px, py, ZOOM, ZOOM))

    return bg


def get_region_top_lefts(game, char_type):
    """Haritada yan yana duran aynı harf gruplarının sadece en sol-üst karesini döner."""
    visited = set()
    top_lefts = []
    
    for x in range(game.mapW):
        for y in range(game.mapH):
            if game.map[x, y] == char_type and (x, y) not in visited:
                # Yeni bir bölge başlangıcı (BFS ile tamamını ziyaret edelim ki tekrar saymasın)
                queue = [(x, y)]
                visited.add((x, y))
                region_cells = [(x, y)]
                
                head = 0
                while head < len(queue):
                    cx, cy = queue[head]
                    head += 1
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < game.mapW and 0 <= ny < game.mapH:
                            if game.map[nx, ny] == char_type and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
                                region_cells.append((nx, ny))
                
                # Bölgenin en sol üst karesini (önce x'i en küçük, sonra y'si en küçük) bulalım
                sorted_cells = sorted(region_cells, key=lambda p: (p[0], p[1]))
                top_lefts.append(sorted_cells[0])
                
    return top_lefts
def draw_fancy_text(S, grid_x, grid_y, text, text_color=(255, 255, 255), bg_color=(0, 0, 0, 150), padding=6, radius=8):
    """
    Kare koordinatlarına göre kalın, renkli ve arkası oval (yuvarlatılmış) arka planlı yazı yazar.
    """
    # 1. Yazı tipini kalın (Bold) yapıyoruz
    S.font.set_bold(True)
    
    # 2. Metni oluşturuyoruz (Render)
    text_surf = S.font.render(text, True, text_color)
    
    # 3. Kare koordinatlarını (grid) ekrandaki piksel koordinatlarına çeviriyoruz
    # (Not: Kodunuzdaki dönüşüm mantığına göre S.X veya S.ZOOM ile çarpılıyor olabilir, burayı kendi sisteminize göre eşitleyin)
    pixel_x = int(grid_x * ZOOM)
    pixel_y = int(grid_y * ZOOM)
    
    # 4. Oval arka planın boyutlarını yazının boyutuna göre hesaplıyoruz (padding = kenar boşluğu)
    bg_w = text_surf.get_width() + (padding * 2)
    bg_h = text_surf.get_height() + (padding * 2)
    
    # Yazının arka planın tam ortasına gelmesi için ofsetler
    bg_x = pixel_x - padding
    bg_y = pixel_y - padding
    
    # 5. Saydam arka plan desteği için geçici bir yüzey (Surface) oluşturuyoruz
    # bg_color içindeki 4. değer (Alfa) saydamlığı belirler (0: Tam saydam, 255: Opak)
    bg_surf = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
    
    # 6. Oval dikdörtgeni bu geçici yüzeye çiziyoruz (radius = oval köşe yarıçapı)
    pygame.draw.rect(bg_surf, bg_color, (0, 0, bg_w, bg_h), border_radius=radius)
    
    # 7. Önce arka planı, sonra yazıyı ekrana basıyoruz (Blit)
    S.screen.blit(bg_surf, (bg_x, bg_y))
    S.screen.blit(text_surf, (pixel_x, pixel_y))
    
    # İşi bittikten sonra fontu eski normal haline döndürüyoruz ki diğer yazılar kalın olmasın
    S.font.set_bold(False)

def draw_map(game, S, background, trucks, spawn_timer):
    S.screen.blit(background, (0, 0))

    # ── DESSIN DE LA GRANDE SCIERIE (S) ──
    if game.scieries:
        sorted_s = sorted(game.scieries, key=lambda p: (p[0], p[1]))
        first_s = sorted_s[0]
        S.drawLargeScierieSprite(first_s[0], first_s[1])

    # ── DESSIN DE LA GRANDE FABRIQUE DE MEUBLES (M) ──
    if game.factories:
        factories_sorted = sorted(game.factories, key=lambda p: (p[0], p[1]))
        exact_top_left_m = factories_sorted[0]
        S.drawLargeFactorySprite(exact_top_left_m[0], exact_top_left_m[1])

    # ── RENDU DES VILLES (V) ──
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

    # ── RENDU DES FORETS (F) ──
    for (x, y), wood in game.forests.items():
        if S.pintree_img and S.deadtree_img:
            if wood == WOOD_PER_FOREST:
                S.drawImage(x, y, S.pintree_img)
            elif wood > 0:
                S.drawImage(x, y, S.deadtree_img)
        else:
            if wood == WOOD_PER_FOREST:
                S.drawText(x, y, "🌲", big=False, centered=True)
            elif wood > 0:
                S.drawText(x, y, "🌱", big=False, centered=True)

# ── METİNLER : DAHA DA YUKARI KAYDIRILMIŞ KOORDİNATLAR ──

    # Scierie (S) - 'ty - 0.5' idi, 'ty - 1.2' yaparak bir kare boyundan fazla yukarı taşıdık
    for tx, ty in get_region_top_lefts(game, 'S'):
        draw_fancy_text(S, tx + 0.2, ty - 1.2, "Scierie", 
                        text_color=(255, 255, 255), 
                        bg_color=(30, 30, 30, 200))

    # Fabrique (M) - İzometrik çatı yüksekliğini kurtarmak için 'ty - 1.4' e çekildi
    for tx, ty in get_region_top_lefts(game, 'M'):
        draw_fancy_text(S, tx + 0.1, ty - 1.4, "Fabrique", 
                        text_color=(255, 255, 255), 
                        bg_color=(30, 30, 30, 200))

    # Ormanlar (F) - Ağaçların yapraklarının üzerine, tam tepelerine gelmesi için 'ty - 0.8' yapıldı
    for tx, ty in get_region_top_lefts(game, 'F'):
        if game.forests.get((tx, ty), 0) > 0:
            draw_fancy_text(S, tx + 0.1, ty - 0.8, "Forêt", 
                            text_color=(10, 60, 10), 
                            bg_color=(200, 240, 200, 180))

    # Şehirler (V) - Ev çatılarının yukarısında durması için 'ty - 0.8' yapıldı
    for tx, ty in get_region_top_lefts(game, 'V'):
        draw_fancy_text(S, tx + 0.1, ty - 0.8, "Ville", 
                        text_color=(0, 0, 0), 
                        bg_color=(255, 220, 150, 180))

    # Depolar (D) - Garaj alanının bir kare yukarısına 'ty - 0.8' ile taşındı
    for tx, ty in get_region_top_lefts(game, 'D'):
        draw_fancy_text(S, tx + 0.1, ty - 0.8, "Dépôt", 
                        text_color=(240, 240, 255), 
                        bg_color=(50, 50, 150, 180))

    # ── DESSINER LES TRUCKS ──
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
    # ── draw_map fonksiyonunun içinde, en alttaki S.show()'dan hemen önceye ekleyin ──
    for ft in game.floating_texts:
        ft.draw(S)

    # ── Alt Bar Metnini de Güncelleyelim (Kazanılan Para Göstergesi) ──
    # ── Alt Bar Çizimi ──
    bar_y = S.H - ZOOM
    pygame.draw.rect(S.screen, (25, 25, 25), (0, bar_y, S.W, ZOOM))
    
    total = game.total_furniture_delivered
    remaining = sum(game.forests.values())
    
    if len(trucks) < MAX_TRUCKS:
        timer_text = f" | Nouveau : {max(0.0, spawn_timer):.1f}s"
    else:
        timer_text = " | Garage Plein (Max 5)"

    # Yazıların dikeyde tam ortalanması için gereken Y koordinatı
    text_center_y = bar_y + (ZOOM // 2 - S.font.size("A")[1] // 2)
    current_x = 15  # Sol kenardan başlama mesafesi

    # 1. Hazine İkonunu Çizdirme (Eğer başarıyla yüklendiyse)
    if S.treasure_icon:
        icon_rect = S.treasure_icon.get_rect()
        icon_rect.left = current_x
        icon_rect.centery = bar_y + (ZOOM // 2)
        S.screen.blit(S.treasure_icon, icon_rect.topleft)
        current_x += S.treasure_icon.get_width() + 8  # İkonun genişliği + boşluk kadar sağa kaydır

    # 2. Para Miktarını Yazdırma (Sarı renkle parlaması için rengini değiştirdik)
    money_text = f"Trésor: ${game.money}"
    money_surf = S.font.render(money_text, True, (255, 215, 0)) # Altın sarısı
    S.screen.blit(money_surf, (current_x, text_center_y))
    current_x += money_surf.get_width() + 20  # Para yazısı bittikten sonra diğer metin için boşluk bırak

    # 3. Geri Kalan İstatistikleri Yazdırma (Beyaz renk)
    stats_text = f"|  Livraisons: {total}  |  Bois: {remaining}  |  Camions: {len(trucks)}/{MAX_TRUCKS}{timer_text}"
    stats_surf = S.font.render(stats_text, True, (240, 240, 240))
    S.screen.blit(stats_surf, (current_x, text_center_y))

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

                    for ft in game.floating_texts:
                        ft.update(dt)
                    game.floating_texts = [ft for ft in game.floating_texts if ft.lifetime > 0]
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