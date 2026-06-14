import sys
import math
import random
import os
import numpy as np
import pygame
from numba import njit

# ── Constants ────────────────────────────────────────────────────────────────

WALKABLE_INIT   = 100
BLOCKED_INIT    = 999
TRUCK_SPEED     = 4.0
LOAD_TIME       = 1.0
UNLOAD_TIME     = 0.8
WOOD_PER_FOREST = 2
MAX_TRUCKS      = 5
DEFAULT_ZOOM    = 45
MIN_ZOOM        = 20

ASSETS_DIR      = "assets"
ZOOM            = DEFAULT_ZOOM

# ── Distance Map using Numba (Wavefront Algorithm) ───────────────────────────

@njit(cache=True)
def compute_distance_map_numba(base_dist, targets_xy, max_iterations, out_dist):
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


# ── Map Layout ───────────────────────────────────────────────────────────────

MAP_TEXT = """
#######################
########          #WWW#
#FFFFFF#    ##### #####
#                     #
#         # #         #
#         # #DD#  #####
# SSS        DD#  MMM #
# SSS         ##  MMM #
#        ####     ### #
# ##     #WW#         #
#        ######### VVV#
####               VVV#
#                     #
#######################
"""


# ── Game Data ────────────────────────────────────────────────────────────────

class GameData:
    def __init__(self, map_text):
        self.map, self.mapW, self.mapH = self._parse(map_text)
        self.base_dist = self._build_base_dist()
        self.game_over = False

        self.forests   = {(x, y): WOOD_PER_FOREST for x in range(self.mapW) for y in range(self.mapH) if self.map[x, y] == 'F'}
        self.scieries  = [(x, y) for x in range(self.mapW) for y in range(self.mapH) if self.map[x, y] == 'S']
        self.factories = [(x, y) for x in range(self.mapW) for y in range(self.mapH) if self.map[x, y] == 'M']
        self.cities    = {(x, y): 0 for x in range(self.mapW) for y in range(self.mapH) if self.map[x, y] == 'V'}
        self.depots    = [(x, y) for x in range(self.mapW) for y in range(self.mapH) if self.map[x, y] == 'D']

        self.total_furniture_delivered = 0
        self.money       = 20
        self.truck_price = 10
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
        base = np.full((self.mapW, self.mapH), BLOCKED_INIT, dtype=np.int32)
        for x in range(self.mapW):
            for y in range(self.mapH):
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
            if self.forests[pos] == 0:
                self.map[pos[0], pos[1]] = ' '
            return True
        return False

    def forest_exhausted(self, pos):
        return self.forests.get(pos, 0) <= 0

    def deliver(self, pos):
        if pos in self.cities:
            self.cities[pos] += 1
            self.total_furniture_delivered += 1
            self.money += 9
            self.floating_texts.append(FloatingText(pos[0], pos[1], "+$9"))
            self.money -= 5
            self.floating_texts.append(FloatingText(pos[0], pos[1] + 0.4, "Gas: -$5", color=(255, 50, 50)))


# ── Screen and Graphics ───────────────────────────────────────────────────────

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
        pygame.display.set_caption("Transport Tycoon - Logistics")
        self.clock = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Arial", max(12, int(ZOOM * 0.35)), bold=True)
        self.font_s = pygame.font.SysFont("Arial", max(10, int(ZOOM * 0.30)))

        def create_fallback(w, h, color):
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            surf.fill(color)
            return surf

        self.truck_images = []
        for name in ["car-truck2.png", "car-truck3.png", "car-truck4.png", "car-truck5.png"]:
            path = os.path.join(ASSETS_DIR, name)
            try:
                img = pygame.image.load(path).convert_alpha()
                orig_w, orig_h = img.get_size()
                target_w = int(ZOOM * 0.55)
                target_h = int(target_w * (orig_h / orig_w))
                if target_h > int(ZOOM * 0.85):
                    target_h = int(ZOOM * 0.85)
                    target_w = int(target_h * (orig_w / orig_h))
                self.truck_images.append(pygame.transform.scale(img, (target_w, target_h)))
            except Exception:
                self.truck_images.append(create_fallback(int(ZOOM * 0.5), int(ZOOM * 0.8), (200, 50, 50)))

        try:
            img = pygame.image.load(os.path.join(ASSETS_DIR, "BrickHouse.png")).convert_alpha()
            self.house_img = pygame.transform.smoothscale(img, (3 * ZOOM, 2 * ZOOM))
        except Exception:
            self.house_img = None

        try:
            img = pygame.image.load(os.path.join(ASSETS_DIR, "BrickTiles.png")).convert_alpha()
            self.tile_img = pygame.transform.scale(img, (ZOOM, ZOOM))
        except Exception:
            self.tile_img = None

        try:
            img = pygame.image.load(os.path.join(ASSETS_DIR, "grass.png")).convert_alpha()
            self.grass_img = pygame.transform.smoothscale(img, (ZOOM, ZOOM))
        except Exception:
            self.grass_img = None

        try:
            img = pygame.image.load(os.path.join(ASSETS_DIR, "water.png")).convert_alpha()
            self.water_img = pygame.transform.smoothscale(img, (ZOOM, ZOOM))
        except Exception:
            self.water_img = None

        try:
            self.factory_asset = pygame.image.load(os.path.join(ASSETS_DIR, "factory.png")).convert_alpha()
            self.factory_asset = pygame.transform.scale(self.factory_asset, (int(ZOOM * 3), int(ZOOM * 2.5)))
        except Exception:
            self.factory_asset = create_fallback(int(ZOOM * 3), int(ZOOM * 2), (120, 120, 120))

        try:
            self.office_asset = pygame.image.load(os.path.join(ASSETS_DIR, "isometric_office_5.png")).convert_alpha()
            self.office_asset = pygame.transform.scale(self.office_asset, (int(ZOOM * 3), int(ZOOM * 2.5)))
        except Exception:
            self.office_asset = create_fallback(int(ZOOM * 3), int(ZOOM * 2), (70, 90, 120))

        try:
            self.pintree_img  = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'pintree.png')).convert_alpha(), (ZOOM, ZOOM))
            self.deadtree_img = pygame.transform.scale(pygame.image.load(os.path.join(ASSETS_DIR, 'deadtree.png')).convert_alpha(), (ZOOM, ZOOM))
        except Exception:
            self.pintree_img  = create_fallback(ZOOM, ZOOM, (35, 150, 55))
            self.deadtree_img = create_fallback(ZOOM, ZOOM, (130, 100, 70))

        try:
            img = pygame.image.load(os.path.join(ASSETS_DIR, "treasure.png")).convert_alpha()
            self.treasure_icon = pygame.transform.scale(img, (int(ZOOM * 0.6), int(ZOOM * 0.6)))
        except Exception:
            self.treasure_icon = None

    def grid_to_screen(self, x, y):
        return int(x * ZOOM), int(y * ZOOM)

    def drawTruckSprite(self, x, y, dx, dy, img_index, cargo_type):
        cx, cy = int(x * ZOOM), int(y * ZOOM)
        base_img = self.truck_images[img_index % len(self.truck_images)]

        if   dy < 0: angle = 0
        elif dy > 0: angle = 180
        elif dx > 0: angle = 270
        else:        angle = 90

        rotated_img = pygame.transform.rotate(base_img, angle)
        rect = rotated_img.get_rect(center=(cx, cy))
        self.screen.blit(rotated_img, rect.topleft)

        if cargo_type == "wood":
            pygame.draw.circle(self.screen, (139, 90, 43), (cx, cy), 4)
        elif cargo_type == "planks":
            pygame.draw.circle(self.screen, (222, 184, 135), (cx, cy), 4)
        elif cargo_type == "furniture":
            pygame.draw.circle(self.screen, (255, 140, 0), (cx, cy), 4)

    def drawLargeScierieSprite(self, x, y):
        if self.factory_asset:
            sx, sy = self.grid_to_screen(x, y)
            rect = self.factory_asset.get_rect()
            rect.bottomleft = (sx, sy + (ZOOM * 2))
            self.screen.blit(self.factory_asset, rect.topleft)

    def drawLargeFactorySprite(self, x, y):
        if self.office_asset:
            sx, sy = self.grid_to_screen(x, y)
            rect = self.office_asset.get_rect()
            rect.topleft = (sx, sy - int(ZOOM * 0.5))
            self.screen.blit(self.office_asset, rect.topleft)

    def show(self):
        pygame.display.flip()


# ── Floating Text ─────────────────────────────────────────────────────────────

class FloatingText:
    def __init__(self, grid_x, grid_y, text, color=(255, 215, 0)):
        self.x        = grid_x + 0.5
        self.y        = grid_y
        self.text     = text
        self.color    = color
        self.lifetime = 2.0
        self.vel_y    = -1.0

    def update(self, dt):
        self.y        += self.vel_y * dt
        self.lifetime -= dt

    def draw(self, S):
        if self.lifetime <= 0:
            return
        alpha = max(0, min(255, int(self.lifetime * 255)))
        S.font.set_bold(True)
        surf = S.font.render(self.text, True, self.color)
        alpha_surf = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        alpha_surf.fill((255, 255, 255, alpha))
        surf.blit(alpha_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        sx = int(self.x * ZOOM) - surf.get_width() // 2
        sy = int(self.y * ZOOM)
        S.screen.blit(surf, (sx, sy))
        S.font.set_bold(False)


# ── Truck ─────────────────────────────────────────────────────────────────────

class Truck:
    _id_counter = 0

    def __init__(self, game: GameData, depot_pos):
        Truck._id_counter += 1
        self.tid       = Truck._id_counter
        self.img_index = (self.tid - 1) % 4

        self.cx, self.cy = depot_pos
        self.nx, self.ny = depot_pos
        self.x = self.cx + 0.5
        self.y = self.cy + 0.5

        self.dir   = (0.0, -1.0)
        self.speed = TRUCK_SPEED + random.uniform(-0.2, 0.2)

        self.cargo      = None
        self.state      = "seeking_forest"
        self.timer      = 0.0
        self.target_pos = None

        self.dist = np.empty_like(game.base_dist)
        self._pick_forest_target(game)

    # ── Target selection ──────────────────────────────────────────────────────

    def _pick_forest_target(self, game):
        avail = game.available_forests()
        if not avail:
            # No forests left → end game
            game.game_over = True
            return

        # Pick forest with best wood-per-distance ratio
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
        self.target_pos = min(cities, key=lambda p: abs(p[0]-self.cx) + abs(p[1]-self.cy))
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

    # ── Pathfinding with deadlock prevention ──────────────────────────────────

    def _next_grid_step(self, game, all_trucks):
        # Cases réservées par les autres camions (position actuelle ET prochaine)
        reserved = set()
        for t in all_trucks:
            if t.tid != self.tid:
                reserved.add((t.cx, t.cy))
                reserved.add((t.nx, t.ny))

        # Voisins walkables triés par distance
        cands = []
        for ddx, ddy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = self.cx + ddx, self.cy + ddy
            if not self._is_walkable(game, nx, ny):
                continue
            nd = self.dist[nx, ny]
            if nd < BLOCKED_INIT:
                cands.append((nd, nx, ny))

        if not cands:
            self.nx, self.ny = self.cx, self.cy
            return

        cands.sort(key=lambda item: item[0])

        # 1. Meilleure case libre
        for nd, nx, ny in cands:
            if (nx, ny) not in reserved:
                self.nx, self.ny = nx, ny
                return

        # 2. Toutes bloquées → camion avec le plus petit tid a priorité
        # Le camion perdant attend sur place au lieu de foncer dans l'autre
        best_nd, best_nx, best_ny = cands[0]
        blocker_tid = None
        for t in all_trucks:
            if t.tid != self.tid and (t.cx, t.cy) == (best_nx, best_ny):
                blocker_tid = t.tid
                break
            if t.tid != self.tid and (t.nx, t.ny) == (best_nx, best_ny):
                blocker_tid = t.tid
                break

        # Si notre tid est plus grand, on cède le passage
        if blocker_tid is not None and self.tid > blocker_tid:
            self.nx, self.ny = self.cx, self.cy  # on attend
            return

        # Sinon on prend une case alternative
        for ddx, ddy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = self.cx + ddx, self.cy + ddy
            if self._is_walkable(game, nx, ny) and (nx, ny) not in reserved:
                self.nx, self.ny = nx, ny
                return

        # Vraiment bloqué → on attend
        self.nx, self.ny = self.cx, self.cy

    def update(self, game: GameData, all_trucks, dt):

        # Idle: wait for a forest to become available
        if self.state == "idle":
            if game.available_forests():
                self._pick_forest_target(game)
            return

        # Loading / Unloading: truck stays still until timer expires
        if self.state in ("loading", "unloading"):
            self.timer -= dt
            if self.timer > 0:
                return

            if self.state == "loading":
                if self.cargo == "wood":
                    self._pick_scierie_target(game)
                elif self.cargo == "planks":
                    self._pick_factory_target(game)
                elif self.cargo == "furniture":
                    self._pick_city_target(game)

            elif self.state == "unloading":
                if self.cargo == "wood":
                    self.cargo = "planks"
                    self._pick_factory_target(game)
                elif self.cargo == "planks":
                    self.cargo = "furniture"
                    self._pick_city_target(game)
                elif self.cargo == "furniture":
                    game.deliver(self.target_pos)
                    self.cargo = None
                    self._pick_forest_target(game)
            return

        # Arrived at current grid cell → decide next action
        if self.cx == self.nx and self.cy == self.ny:
            if self.target_pos:
                tx, ty = self.target_pos
                dist_to_target = abs(tx - self.cx) + abs(ty - self.cy)
                if dist_to_target <= 1:
                    if self.state == "seeking_forest":
                        if not game.forest_exhausted(self.target_pos):
                            game.harvest(self.target_pos)
                            self.cargo = "wood"
                            self.state = "loading"
                            self.timer = LOAD_TIME
                        else:
                            self._pick_forest_target(game)
                        return
                    elif self.state in ("seeking_scierie", "seeking_factory", "seeking_city"):
                        self.state = "unloading"
                        self.timer = UNLOAD_TIME
                        return

            # Not at target yet → compute next step
            self._next_grid_step(game, all_trucks)

# Smooth physical movement toward next grid cell
        target_x = self.nx + 0.5
        target_y = self.ny + 0.5
        vx = target_x - self.x
        vy = target_y - self.y
        dist = math.hypot(vx, vy)

        step = self.speed * dt
        if step >= dist:
            self.x  = target_x
            self.y  = target_y
            self.cx = self.nx
            self.cy = self.ny
        else:
            if dist > 0:
                self.dir = (vx / dist, vy / dist)
            self.x += self.dir[0] * step
            self.y += self.dir[1] * step

        # Anti ping-pong : si deux camions visent la case de l'autre, le plus grand tid recule
        for t in all_trucks:
            if t.tid != self.tid:
                # Ils se visent mutuellement
                if (self.nx, self.ny) == (t.cx, t.cy) and (t.nx, t.ny) == (self.cx, self.cy):
                    if self.tid > t.tid:
                        # On recule sur notre case actuelle
                        self.nx, self.ny = self.cx, self.cy
                        self.x = self.cx + 0.5
                        self.y = self.cy + 0.5
                        break

    def draw(self, S):
        S.drawTruckSprite(self.x, self.y, self.dir[0], self.dir[1], self.img_index, self.cargo)


# ── Background and Map Drawing ────────────────────────────────────────────────

COLOR_ROAD = (235, 230, 225)
COLOR_WALL = (35, 110, 55)


def build_background(game, S):
    bg = pygame.Surface((S.W, S.H))

    if S.tile_img:
        for x in range(game.mapW):
            for y in range(game.mapH):
                bg.blit(S.tile_img, (x * ZOOM, y * ZOOM))
    else:
        bg.fill(COLOR_ROAD)

    house_drawn = False
    for x in range(game.mapW):
        for y in range(game.mapH):
            char = game.map[x, y]
            px = x * ZOOM
            py = y * ZOOM

            if char == '#':
                if S.grass_img: bg.blit(S.grass_img, (px, py))
                else: pygame.draw.rect(bg, COLOR_WALL, (px, py, ZOOM, ZOOM))
            elif char == 'W':
                if S.water_img: bg.blit(S.water_img, (px, py))
                else: pygame.draw.rect(bg, (0, 100, 200), (px, py, ZOOM, ZOOM))
            elif char == 'V':
                if not house_drawn and S.house_img:
                    bg.blit(S.house_img, (px, py))
                    house_drawn = True
                elif not S.house_img:
                    pygame.draw.rect(bg, (200, 100, 50), (px, py, ZOOM, ZOOM))
    return bg


def draw_map(game, S, background, trucks):
    S.screen.blit(background, (0, 0))

    if game.scieries:
        sorted_s = sorted(game.scieries, key=lambda p: (p[0], p[1]))
        S.drawLargeScierieSprite(sorted_s[0][0], sorted_s[0][1])

    if game.factories:
        factories_sorted = sorted(game.factories, key=lambda p: (p[0], p[1]))
        S.drawLargeFactorySprite(factories_sorted[0][0], factories_sorted[0][1])

    # Draw trees: pine = full, dead = half depleted
    for (x, y), wood in game.forests.items():
        if wood == 2:
            if S.pintree_img:  S.screen.blit(S.pintree_img,  (x * ZOOM, y * ZOOM))
        elif wood == 1:
            if S.deadtree_img: S.screen.blit(S.deadtree_img, (x * ZOOM, y * ZOOM))

    for t in trucks:
        t.draw(S)

    # Bottom UI panel
    panel_y = game.mapH * ZOOM
    pygame.draw.rect(S.screen, (30, 30, 40), (0, panel_y, S.W, ZOOM))

    if S.treasure_icon:
        S.screen.blit(S.treasure_icon, (10, panel_y + int(ZOOM * 0.2)))

    ui_text = f"Money: ${game.money}  |  Trucks: {len(trucks)}/{MAX_TRUCKS}"
    text_surf = S.font.render(ui_text, True, (255, 215, 0))
    S.screen.blit(text_surf, (int(2 * ZOOM), panel_y + int(ZOOM * 0.2)))

    # Buy truck button
    btn_x, btn_y, btn_w, btn_h = S.W - 150, panel_y + 5, 145, ZOOM - 10

    if len(trucks) >= MAX_TRUCKS:
        btn_color = (80, 40, 40)
        btn_text  = "MAX TRUCKS"
    elif game.money < game.truck_price:
        btn_color = (100, 100, 100)
        btn_text  = f"Buy Truck (${game.truck_price})"
    else:
        btn_color = (50, 180, 70)
        btn_text  = f"Buy Truck (${game.truck_price})"

    pygame.draw.rect(S.screen, btn_color, (btn_x, btn_y, btn_w, btn_h), border_radius=5)
    font_surf = S.font_s.render(btn_text, True, (255, 255, 255))
    S.screen.blit(font_surf, (btn_x + (btn_w - font_surf.get_width()) // 2, btn_y + (btn_h - font_surf.get_height()) // 2))

    # Game over overlay
    if game.game_over:
        overlay = pygame.Surface((S.W, S.H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        S.screen.blit(overlay, (0, 0))
        big_font = pygame.font.SysFont("Arial", int(ZOOM * 1.1), bold=True)

        # Different message depending on the reason
        if game.money < 0:
            message = "BANKRUPT! GAME OVER"
        else:
            message = "ALL FORESTS DEPLETED — GAME OVER"

        go_surf = big_font.render(message, True, (255, 50, 50))
        S.screen.blit(go_surf, ((S.W - go_surf.get_width()) // 2, (S.H - go_surf.get_height()) // 2))


# ── Main Game Loop ────────────────────────────────────────────────────────────

def main():
    pygame.init()

    pygame.mixer.init()
    music_name = "music.mp3"
    if not os.path.exists(music_name):
        music_name = os.path.join(ASSETS_DIR, "music.mp3")

    if os.path.exists(music_name):
        try:
            pygame.mixer.music.load(music_name)
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f"Music error: {e}")

    game       = GameData(MAP_TEXT)
    S          = Screen(game.mapW, game.mapH)
    background = build_background(game, S)

    trucks = []
    if game.depots:
        trucks.append(Truck(game, game.depots[0]))

    running = True
    while running:
        dt = S.clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and not game.game_over:
                mx, my = pygame.mouse.get_pos()
                panel_y = game.mapH * ZOOM
                btn_x, btn_y, btn_w, btn_h = S.W - 150, panel_y + 5, 145, ZOOM - 10

                if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                    if len(trucks) < MAX_TRUCKS and game.money >= game.truck_price:
                        game.money -= game.truck_price
                        game.truck_price *= 2
                        trucks.append(Truck(game, game.depots[0]))

        if not game.game_over:
            if game.money < 0:
                game.game_over = True
                pygame.mixer.music.stop()

            for t in trucks:
                t.update(game, trucks, dt)

            for ft in game.floating_texts[:]:
                ft.update(dt)
                if ft.lifetime <= 0:
                    game.floating_texts.remove(ft)

        draw_map(game, S, background, trucks)
        for ft in game.floating_texts:
            ft.draw(S)

        S.show()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()