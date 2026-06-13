"""
Transport Tycoon - Camion Simulator
====================================
Kamyonlar ormandan odun alir, sehre tasir.
Sehir hucreleri teslimat yapildikca yesile döner.

Harita sembolleri:
  ' '  = yol (kamyon gecebilir)
  '#'  = engel / duvar
  'F'  = orman (odun kaynagi)
  'V'  = ville / sehir (hedef)
  'D'  = depo (kamyon baslangici)
"""

import sys
import math
import random
import numpy as np
import pygame
from numba import njit


# ── Sabitler ────────────────────────────────────────────────────────────────

WALKABLE_INIT   = 100
BLOCKED_INIT    = 999
TRUCK_SPEED     = 3.0        # hücre / saniye
LOAD_TIME       = 1.2        # saniye (orman'da bekleme)
UNLOAD_TIME     = 0.8        # saniye (sehirde boslama)
WOOD_PER_FOREST = 5          # bir orman hücresinin toplam odunu
NUM_TRUCKS      = 4
DEFAULT_ZOOM    = 48
MIN_ZOOM        = 14


# ── Numba distance map ───────────────────────────────────────────────────────

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


# ── Harita ──────────────────────────────────────────────────────────────────

# Her karakter = 1 hücre.
# F = orman (sol üst köşe)    V = ville/sehir (sag alt köşe)
# D = depo (orta)             # = duvar     bosluk = yol
MAP_TEXT = """
##############################
#FFFFFF                      #
#FFFFFF                      #
#FFFFFF                      #
#FFFFFF                      #
#FFFFFF                      #
#                            #
#                            #
#              DD            #
#              DD            #
#                            #
#                            #
#                    VVVVVV  #
#                    VVVVVV  #
#                    VVVVVV  #
#                    VVVVVV  #
#                    VVVVVV  #
##############################
"""

# Harita yorumlayici
class GameData:
    def __init__(self, map_text):
        self.map, self.mapW, self.mapH = self._parse(map_text)
        self.base_dist = self._build_base_dist()

        # Orman hücreleri: {(x,y): wood_left}
        self.forests = {}
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'F':
                    self.forests[(x, y)] = WOOD_PER_FOREST

        # Şehir hücreleri: {(x,y): delivered_count}
        self.cities = {}
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'V':
                    self.cities[(x, y)] = 0

        # Depo koordinatları
        self.depots = []
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'D':
                    self.depots.append((x, y))

        self.total_wood_delivered = 0

    def _parse(self, text):
        rows = text.strip().splitlines()
        rows = [line for line in rows]
        maxlen = max(len(r) for r in rows)
        rows = [r.ljust(maxlen) for r in rows]
        rows.reverse()

        # Her 3 karakterde bir hücre: " X " veya "XX "
        # Aslında haritayı karakter bazlı okuyalım, boşluklar yol
        # Daha basit: her karakter bir hücre (boşluk = yol)
        # Ama haritada 2 karakter arası boşluk var, bunu parse edelim
        parsed = []
        for row in rows:
            cells = []
            i = 0
            while i < len(row):
                c = row[i]
                if c == '#':
                    cells.append('#')
                    i += 1
                elif c == ' ':
                    # İki boşluk arası padding mi yoksa gerçek yol mu?
                    cells.append(' ')
                    i += 1
                elif c in ('F', 'V', 'D'):
                    cells.append(c)
                    i += 1
                else:
                    cells.append(' ')
                    i += 1
            parsed.append(cells)

        # Uzunluk eşitle
        maxw = max(len(r) for r in parsed)
        for r in parsed:
            while len(r) < maxw:
                r.append(' ')

        arr = np.array(parsed, dtype='U1').T
        w, h = arr.shape
        return arr, w, h

    def _build_base_dist(self):
        base = np.full((self.mapW, self.mapH), BLOCKED_INIT, dtype=np.int32)
        for x in range(self.mapW):
            for y in range(self.mapH):
                c = self.map[x, y]
                if c in (' ', 'D', 'F', 'V'):   # kamyon geçebilir
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

    def available_cities(self):
        return list(self.cities.keys())

    def harvest(self, pos):
        """Ormandan 1 odun al. Bittiyse False döner."""
        if self.forests.get(pos, 0) > 0:
            self.forests[pos] -= 1
            return True
        return False

    def deliver(self, pos):
        """Şehre teslimat yap."""
        if pos in self.cities:
            self.cities[pos] += 1
            self.total_wood_delivered += 1

    def forest_exhausted(self, pos):
        return self.forests.get(pos, 0) <= 0


# ── Ekran ────────────────────────────────────────────────────────────────────

ZOOM = DEFAULT_ZOOM

class Screen:
    def __init__(self, nx, ny):
        pygame.display.init()
        pygame.font.init()
        global ZOOM
        info = pygame.display.Info()
        max_w = max(320, info.current_w - 80)
        max_h = max(240, info.current_h - 140)
        ZOOM = max(MIN_ZOOM, min(DEFAULT_ZOOM, max_w // nx, max_h // (ny + 2)))

        self.W = nx * ZOOM
        self.H = (ny + 2) * ZOOM
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Transport Tycoon - Camion Sim")
        self.clock = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Arial", max(12, int(ZOOM * 0.55)), bold=True)
        self.font_s = pygame.font.SysFont("Arial", max(9,  int(ZOOM * 0.35)))

    def grid_to_screen(self, x, y):
        sx = x * ZOOM
        sy = self.H - (y + 2) * ZOOM   # +2: status bar altta
        return sx, sy

    def drawRect(self, x, y, w=1, h=1, color=(0,0,0), border=0):
        sx, sy = self.grid_to_screen(x, y)
        pygame.draw.rect(self.screen, color,
                         (sx, sy - h*ZOOM + ZOOM, w*ZOOM, h*ZOOM), border)

    def drawCircle(self, x, y, r, color):
        sx, sy = self.grid_to_screen(x, y)
        cx = sx + ZOOM // 2
        cy = sy - ZOOM // 2
        pygame.draw.circle(self.screen, color, (cx, cy), int(r * ZOOM))

    def drawText(self, x, y, txt, color=(255,255,255), big=True, centered=False):
        font = self.font if big else self.font_s
        sx, sy = self.grid_to_screen(x, y)
        surf = font.render(str(txt), True, color)
        ox = -surf.get_width()//2  if centered else 0
        oy = -surf.get_height()//2
        self.screen.blit(surf, (sx + ox, sy + oy))

    def drawTruck(self, x, y, dx, dy, color, loaded):
        """Kamyonu belirtilen grid koordinatında çiz."""
        sx, sy = self.grid_to_screen(x, y)
        cx = sx + ZOOM // 2
        cy = sy - ZOOM // 2

        # Gövde
        body_w = int(ZOOM * 0.72)
        body_h = int(ZOOM * 0.44)
        body_rect = pygame.Rect(cx - body_w//2, cy - body_h//2, body_w, body_h)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=4)
        pygame.draw.rect(self.screen, (30, 30, 30), body_rect, 2, border_radius=4)

        # Kabin (yön bağlı)
        cab_w = int(ZOOM * 0.24)
        cab_h = int(ZOOM * 0.34)
        # Yönü belirle
        if abs(dx) >= abs(dy):
            cab_x = cx + (body_w//2 - cab_w) if dx > 0 else cx - body_w//2
        else:
            cab_x = cx - cab_w//2
        cab_rect = pygame.Rect(cab_x, cy - cab_h//2, cab_w, cab_h)
        pygame.draw.rect(self.screen, (60, 60, 90), cab_rect, border_radius=3)

        # Yük göstergesi
        if loaded:
            cargo_w = int(ZOOM * 0.42)
            cargo_h = int(ZOOM * 0.22)
            cargo_rect = pygame.Rect(cx - cargo_w//2, cy - body_h//2 - cargo_h + 2,
                                     cargo_w, cargo_h)
            pygame.draw.rect(self.screen, (160, 100, 40), cargo_rect, border_radius=2)
            pygame.draw.rect(self.screen, (80, 50, 20), cargo_rect, 1, border_radius=2)

        # Tekerlekler
        wheel_r = max(3, int(ZOOM * 0.10))
        for wx in [cx - body_w//3, cx + body_w//3]:
            pygame.draw.circle(self.screen, (20, 20, 20),
                               (wx, cy + body_h//2 + wheel_r - 2), wheel_r)

    def show(self):
        pygame.display.flip()


# ── Kamyon ───────────────────────────────────────────────────────────────────

TRUCK_COLORS = [
    (220, 60,  60),   # kırmızı
    (60,  120, 220),  # mavi
    (220, 180, 40),   # sarı
    (60,  200, 100),  # yeşil
]

class Truck:
    _id_counter = 0

    def __init__(self, game: GameData, depot_pos):
        Truck._id_counter += 1
        self.tid  = Truck._id_counter
        self.color = TRUCK_COLORS[(self.tid - 1) % len(TRUCK_COLORS)]

        dx, dy = depot_pos
        self.x = dx + 0.5
        self.y = dy + 0.5
        self.dir  = (1.0, 0.0)
        self.speed = TRUCK_SPEED + random.uniform(-0.4, 0.4)

        self.loaded  = False          # odun taşıyor mu?
        self.state   = "seeking_forest"   # seeking_forest | loading | seeking_city | unloading | idle
        self.timer   = 0.0
        self.target_pos = None        # hedef hücre (x,y)

        self.dist = np.empty_like(game.base_dist)
        self.deliveries = 0

        self._pick_forest_target(game)

    # ── Hedef seçimi ────────────────────────────────────────────────────────

    def _pick_forest_target(self, game):
        """En iyi ormanı seç: wood / mesafe oranı maksimum."""
        avail = game.available_forests()
        if not avail:
            self.state = "idle"
            self.target_pos = None
            return

        cx, cy = int(self.x), int(self.y)
        best, best_score = None, -1

        for (fx, fy), wood in avail:
            dist = abs(fx - cx) + abs(fy - cy) or 1
            score = wood / dist
            if score > best_score:
                best_score = score
                best = (fx, fy)

        self.target_pos = best
        self.state = "seeking_forest"
        self._refresh_dist(game)

    def _pick_city_target(self, game):
        """En yakın şehri seç."""
        cities = game.available_cities()
        if not cities:
            self.state = "idle"
            self.target_pos = None
            return

        cx, cy = int(self.x), int(self.y)
        best = min(cities, key=lambda p: abs(p[0]-cx) + abs(p[1]-cy))
        self.target_pos = best
        self.state = "seeking_city"
        self._refresh_dist(game)

    def _refresh_dist(self, game):
        if self.target_pos is None:
            game.computeDistanceMap(
                np.empty((0, 2), dtype=np.int32), self.dist)
            return
        tx, ty = self.target_pos
        targets = np.array([[tx, ty]], dtype=np.int32)
        game.computeDistanceMap(targets, self.dist)

    # ── Hareket ─────────────────────────────────────────────────────────────

    def _grid(self):
        return int(self.x), int(self.y)

    def _is_walkable(self, game, x, y):
        if not (0 <= x < game.mapW and 0 <= y < game.mapH):
            return False
        return game.map[x, y] not in ('#',)

    def _best_neighbor(self, game):
        cx, cy = self._grid()
        cur = self.dist[cx, cy]
        best_d = cur
        cands = []
        for ddx, ddy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cx+ddx, cy+ddy
            if not self._is_walkable(game, nx, ny):
                continue
            nd = self.dist[nx, ny]
            if nd < best_d:
                best_d = nd
                cands = [(nx, ny)]
            elif nd == best_d:
                cands.append((nx, ny))
        return random.choice(cands) if cands else None

    def _adjacent_to_target(self):
        if self.target_pos is None:
            return False
        tx, ty = self.target_pos
        cx, cy = self._grid()
        return abs(tx - cx) + abs(ty - cy) <= 1

    def _move_toward_target(self, game, dt):
        cx, cy = self._grid()

        # Hedefe yeterince yakın mı?
        if self.target_pos and self._adjacent_to_target():
            return True   # vardi

        nb = self._best_neighbor(game)
        if nb is None:
            return False

        nx, ny = nb
        tx, ty = nx + 0.5, ny + 0.5
        vx, vy = tx - self.x, ty - self.y
        d = math.hypot(vx, vy)
        if d < 1e-6:
            self.x, self.y = tx, ty
            return False

        self.dir = (vx/d, vy/d)
        step = self.speed * dt
        if step >= d:
            self.x, self.y = tx, ty
        else:
            self.x += self.dir[0] * step
            self.y += self.dir[1] * step
        return False

    # ── Ana güncelleme ───────────────────────────────────────────────────────

    def update(self, game: GameData, dt):
        if self.state == "idle":
            # Yeni orman çıktı mı diye bak
            if game.available_forests():
                self._pick_forest_target(game)
            return

        if self.state == "loading":
            self.timer -= dt
            if self.timer <= 0:
                self.loaded = True
                self._pick_city_target(game)
            return

        if self.state == "unloading":
            self.timer -= dt
            if self.timer <= 0:
                game.deliver(self.target_pos)
                self.deliveries += 1
                self.loaded = False
                self._pick_forest_target(game)
            return

        # Hareket halinde
        arrived = self._move_toward_target(game, dt)

        if arrived or self._adjacent_to_target():
            if self.state == "seeking_forest":
                fp = self.target_pos
                if fp and not game.forest_exhausted(fp):
                    game.harvest(fp)
                    self.state = "loading"
                    self.timer = LOAD_TIME
                else:
                    # Başka orman bul
                    self._pick_forest_target(game)

            elif self.state == "seeking_city":
                self.state = "unloading"
                self.timer = UNLOAD_TIME

    # ── Çizim ────────────────────────────────────────────────────────────────

    def draw(self, S: Screen):
        S.drawTruck(self.x, self.y, self.dir[0], self.dir[1],
                    self.color, self.loaded)


# ── Arkaplan ─────────────────────────────────────────────────────────────────

COLOR_ROAD    = (210, 200, 185)
COLOR_WALL    = (80,  80,  80)
COLOR_FOREST  = (34,  139, 34)
COLOR_FOREST_EMPTY = (120, 90, 60)
COLOR_CITY_BASE = (180, 160, 130)
COLOR_CITY_DONE = (80, 200, 80)
COLOR_DEPOT   = (160, 160, 200)
COLOR_GRID    = (190, 180, 165)

def build_background(game: GameData, S: Screen):
    surf = pygame.Surface((S.W, S.H))
    surf.fill((50, 50, 50))

    for x in range(game.mapW):
        for y in range(game.mapH):
            c = game.map[x, y]
            if c == '#':
                color = COLOR_WALL
            elif c == 'F':
                wood = game.forests.get((x,y), WOOD_PER_FOREST)
                if wood > 0:
                    ratio = wood / WOOD_PER_FOREST
                    r = int(COLOR_FOREST_EMPTY[0] + ratio*(COLOR_FOREST[0]-COLOR_FOREST_EMPTY[0]))
                    g = int(COLOR_FOREST_EMPTY[1] + ratio*(COLOR_FOREST[1]-COLOR_FOREST_EMPTY[1]))
                    b = int(COLOR_FOREST_EMPTY[2] + ratio*(COLOR_FOREST[2]-COLOR_FOREST_EMPTY[2]))
                    color = (r, g, b)
                else:
                    color = COLOR_FOREST_EMPTY
            elif c == 'V':
                color = COLOR_CITY_BASE
            elif c == 'D':
                color = COLOR_DEPOT
            else:
                color = COLOR_ROAD

            sx, sy = S.grid_to_screen(x, y)
            pygame.draw.rect(surf, color, (sx, sy - ZOOM + ZOOM, ZOOM, ZOOM))
            pygame.draw.rect(surf, COLOR_GRID, (sx, sy - ZOOM + ZOOM, ZOOM, ZOOM), 1)

    return surf

def draw_city_overlay(game: GameData, S: Screen):
    """Teslimat yapılmış şehirleri yeşile boyar."""
    for (x, y), count in game.cities.items():
        if count == 0:
            continue
        ratio = min(1.0, count / 4.0)
        r = int(COLOR_CITY_BASE[0] + ratio*(COLOR_CITY_DONE[0]-COLOR_CITY_BASE[0]))
        g = int(COLOR_CITY_BASE[1] + ratio*(COLOR_CITY_DONE[1]-COLOR_CITY_BASE[1]))
        b = int(COLOR_CITY_BASE[2] + ratio*(COLOR_CITY_DONE[2]-COLOR_CITY_BASE[2]))
        S.drawRect(x, y, color=(r, g, b))
        S.drawRect(x, y, color=(0,0,0), border=1)

def draw_forest_overlay(game: GameData, S: Screen):
    """Ormanların doluluk oranını gösterir."""
    for (x, y), wood in game.forests.items():
        ratio = wood / WOOD_PER_FOREST
        r = int(COLOR_FOREST_EMPTY[0] + ratio*(COLOR_FOREST[0]-COLOR_FOREST_EMPTY[0]))
        g = int(COLOR_FOREST_EMPTY[1] + ratio*(COLOR_FOREST[1]-COLOR_FOREST_EMPTY[1]))
        b = int(COLOR_FOREST_EMPTY[2] + ratio*(COLOR_FOREST[2]-COLOR_FOREST_EMPTY[2]))
        S.drawRect(x, y, color=(r, g, b))
        S.drawRect(x, y, color=(0,0,0), border=1)
        # Ağaç sembolü
        if wood > 0:
            S.drawText(x + 0.5, y + 0.5, "🌲" if wood > 2 else "🌱",
                       color=(0,0,0), big=False, centered=True)

def draw_map(game, S, background, trucks):
    S.screen.blit(background, (0, 0))
    draw_forest_overlay(game, S)
    draw_city_overlay(game, S)

    # Hedef çizgisi (kamyon → hedef)
    for truck in trucks:
        if truck.target_pos and truck.state not in ("loading", "unloading", "idle"):
            sx, sy = S.grid_to_screen(truck.x, truck.y)
            tx, ty = S.grid_to_screen(truck.target_pos[0]+0.5,
                                       truck.target_pos[1]+0.5)
            tcol = (*truck.color, 80)
            pygame.draw.line(S.screen, truck.color,
                             (sx + ZOOM//2, sy - ZOOM//2),
                             (tx + ZOOM//2, ty - ZOOM//2), 1)

    for truck in trucks:
        truck.draw(S)

    # Status bar
    bar_y = S.H - ZOOM
    pygame.draw.rect(S.screen, (20, 20, 20), (0, bar_y, S.W, ZOOM))
    total = game.total_wood_delivered
    remaining = sum(game.forests.values())
    txt = (f"  Teslimat: {total}   |   Ormanda kalan: {remaining}   |"
           f"   Kamyon: {NUM_TRUCKS}   |   SPACE = Pause")
    surf = S.font.render(txt, True, (220, 220, 220))
    S.screen.blit(surf, (8, bar_y + ZOOM//2 - surf.get_height()//2))

    S.show()


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    game = GameData(MAP_TEXT)
    S    = Screen(game.mapW, game.mapH)

    # Kamyonları depo konumlarına yerleştir
    trucks = []
    depots = game.depots if game.depots else [(game.mapW//2, game.mapH//2)]
    for i in range(NUM_TRUCKS):
        dp = depots[i % len(depots)]
        trucks.append(Truck(game, dp))

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
                    # Haritayı sıfırla
                    game = GameData(MAP_TEXT)
                    for t in trucks:
                        dp = depots[t.tid % len(depots)]
                        t.__init__(game, dp)
                    background = build_background(game, S)
            elif event.type == LOGIC_TIMER:
                if not PAUSE:
                    dt = 1.0 / LOGIC_FPS
                    for truck in trucks:
                        truck.update(game, dt)
                    # Arkaplanı orman değişince yenile
                    background = build_background(game, S)

        draw_map(game, S, background, trucks)
        S.clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
