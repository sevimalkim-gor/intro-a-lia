import sys
import math
import random
import numpy as np
import pygame
from numba import njit

# ── Sabitler ────────────────────────────────────────────────────────────────

WALKABLE_INIT   = 100
BLOCKED_INIT    = 999
TRUCK_SPEED     = 4.0        
LOAD_TIME       = 1.0        
UNLOAD_TIME     = 0.8        
WOOD_PER_FOREST = 5          
MAX_TRUCKS      = 5          # En fazla 5 kamyon olabilecek
SPAWN_INTERVAL  = 5.0        # 5 saniyede bir yeni kamyon
DEFAULT_ZOOM    = 45         
MIN_ZOOM        = 20


# ── Numba Mesafe Haritası ───────────────────────────────────────────────────

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


# ── BÜYÜTÜLMÜŞ DEPO HARİTASI (20x12) ────────────────────────────────────────

MAP_TEXT = """
####################
#FFFFFF#           #
#FFFFFF# ##### ### #
###### # #   # # # #
#      # # DD# # # #
# ###### # DD# # # #
# #        ### #   #
# # ########## ### #
# # #            # #
#   # ########## # #
#   ###       #VVVVV#
####################
"""

class GameData:
    def __init__(self, map_text):
        self.map, self.mapW, self.mapH = self._parse(map_text)
        self.base_dist = self._build_base_dist()

        self.forests = {}
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'F':
                    self.forests[(x, y)] = WOOD_PER_FOREST

        self.cities = {}
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'V':
                    self.cities[(x, y)] = 0

        self.depots = []
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'D':
                    self.depots.append((x, y))

        self.total_wood_delivered = 0

    def _parse(self, text):
        rows = [line for line in text.strip().splitlines() if line.strip()]
        maxlen = max(len(r) for r in rows)
        
        parsed = []
        for row in rows:
            cells = []
            for c in row.ljust(maxlen):
                if c in ('#', 'F', 'V', 'D'):
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
            self.total_wood_delivered += 1

    def forest_exhausted(self, pos):
        return self.forests.get(pos, 0) <= 0


# ── Ekran Yönetimi ───────────────────────────────────────────────────────────

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
        pygame.display.set_caption("Transport Tycoon - Zaman Ayarlı Lojistik")
        self.clock = pygame.time.Clock()
        self.font   = pygame.font.SysFont("Arial", max(12, int(ZOOM * 0.45)), bold=True)
        self.font_s = pygame.font.SysFont("Arial", max(10, int(ZOOM * 0.35)))

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

    def drawTruck(self, x, y, dx, dy, color, loaded):
        cx, cy = int(x * ZOOM), int(y * ZOOM)
        body_w = int(ZOOM * 0.65)
        body_h = int(ZOOM * 0.40)
        
        if abs(dy) > abs(dx):
            body_w, body_h = body_h, body_w

        body_rect = pygame.Rect(cx - body_w//2, cy - body_h//2, body_w, body_h)
        pygame.draw.rect(self.screen, color, body_rect, border_radius=3)
        pygame.draw.rect(self.screen, (20, 20, 20), body_rect, 1, border_radius=3)

        cab_size = int(ZOOM * 0.25)
        if abs(dx) >= abs(dy): 
            cab_x = cx + (body_w//2 - cab_size) if dx > 0 else cx - body_w//2
            cab_y = cy - cab_size//2
        else: 
            cab_x = cx - cab_size//2
            cab_y = cy + (body_h//2 - cab_size) if dy > 0 else cy - body_h//2

        pygame.draw.rect(self.screen, (30, 30, 45), (cab_x, cab_y, cab_size, cab_size), border_radius=1)

        if loaded:
            cargo_color = (139, 90, 43)
            if abs(dx) >= abs(dy):
                pygame.draw.rect(self.screen, cargo_color, (cx - 3, cy - body_h//3, 6, int(body_h * 0.6)))
            else:
                pygame.draw.rect(self.screen, cargo_color, (cx - body_w//3, cy - 3, int(body_w * 0.6), 6))

    def show(self):
        pygame.display.flip()


# ── Kamyon Sınıfı ───────────────────────────────────────────────────────────

TRUCK_COLORS = [
    (230, 40, 40),   # Kırmızı
    (40, 110, 230),  # Mavi
    (245, 175, 20),  # Sarı
    (35, 185, 80),   # Yeşil
    (170, 60, 210)   # Mor
]

class Truck:
    _id_counter = 0

    def __init__(self, game: GameData, depot_pos):
        Truck._id_counter += 1
        self.tid  = Truck._id_counter
        self.color = TRUCK_COLORS[(self.tid - 1) % len(TRUCK_COLORS)]

        self.cx, self.cy = depot_pos     
        self.nx, self.ny = depot_pos     
        self.x = self.cx + 0.5
        self.y = self.cy + 0.5
        
        self.dir  = (1.0, 0.0)
        self.speed = TRUCK_SPEED + random.uniform(-0.2, 0.2)

        self.loaded   = False
        self.state    = "seeking_forest" 
        self.timer    = 0.0
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
                self.loaded = True
                self._pick_city_target(game)
            return

        if self.state == "unloading":
            self.timer -= dt
            if self.timer <= 0:
                game.deliver(self.target_pos)
                self.loaded = False
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
        S.drawTruck(self.x, self.y, self.dir[0], self.dir[1], self.color, self.loaded)


# ── Çizim Elemanları ─────────────────────────────────────────────────────────

COLOR_ROAD   = (235, 230, 225)
COLOR_WALL   = (110, 110, 110)   
COLOR_FOREST = (34, 139, 34)
COLOR_FOREST_EMPTY = (150, 125, 100)
COLOR_CITY_BASE = (205, 195, 175)
COLOR_CITY_DONE = (100, 210, 130)
COLOR_DEPOT  = (140, 140, 185)   # Belirgin mavi-gri tonlu büyük depo alanı
COLOR_GRID   = (220, 215, 205)

def build_background(game: GameData, S: Screen):
    surf = pygame.Surface((S.W, S.H))
    surf.fill((30, 30, 30))

    for x in range(game.mapW):
        for y in range(game.mapH):
            c = game.map[x, y]
            if c == '#':
                color = COLOR_WALL
            elif c == 'F':
                wood = game.forests.get((x,y), WOOD_PER_FOREST)
                if wood > 0:
                    ratio = wood / WOOD_PER_FOREST
                    color = (
                        int(COLOR_FOREST_EMPTY[0] + ratio*(COLOR_FOREST[0]-COLOR_FOREST_EMPTY[0])),
                        int(COLOR_FOREST_EMPTY[1] + ratio*(COLOR_FOREST[1]-COLOR_FOREST_EMPTY[1])),
                        int(COLOR_FOREST_EMPTY[2] + ratio*(COLOR_FOREST[2]-COLOR_FOREST_EMPTY[2]))
                    )
                else:
                    color = COLOR_FOREST_EMPTY
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

    for (x, y), wood in game.forests.items():
        if wood > 0:
            S.drawText(x, y, "🌲" if wood > 2 else "🌱", big=False, centered=True)

    for truck in trucks:
        truck.draw(S)

    # Alt Bilgi Çubuğu
    bar_y = S.H - ZOOM
    pygame.draw.rect(S.screen, (25, 25, 25), (0, bar_y, S.W, ZOOM))
    total = game.total_wood_delivered
    remaining = sum(game.forests.values())
    
    # Yeni kamyon sayacı bilgisi
    if len(trucks) < MAX_TRUCKS:
        timer_text = f" | Nouveau camion: {max(0.0, spawn_timer):.1f}s"
    else:
        timer_text = " | Garaj Dolu (Maks 5)"

    txt = f"Livraison: {total}  |  Le reste: {remaining}  |  Camion: {len(trucks)}/{MAX_TRUCKS}{timer_text}"
    surf = S.font.render(txt, True, (240, 240, 240))
    S.screen.blit(surf, (15, bar_y + (ZOOM // 2 - surf.get_height() // 2)))

    S.show()


# ── Ana Döngü ─────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    game = GameData(MAP_TEXT)
    S    = Screen(game.mapW, game.mapH)

    depots = game.depots if game.depots else [(game.mapW//2, game.mapH//2)]
    
    # Oyun başlangıcında sadece 1 kamyon var
    trucks = [Truck(game, depots[0])]
    
    # Spawn (Üretim) zamanlayıcısı
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
                    
                    # Kamyonların Güncellenmesi
                    for truck in trucks:
                        truck.update(game, dt)
                    
                    # Dinamik Kamyon Üretim Mantığı
                    if len(trucks) < MAX_TRUCKS:
                        spawn_timer -= dt
                        if spawn_timer <= 0:
                            # Deponun rastgele bir karesinde yeni kamyon oluştur
                            dp = random.choice(depots)
                            trucks.append(Truck(game, dp))
                            spawn_timer = SPAWN_INTERVAL # Sayacı sıfırla
                    
                    background = build_background(game, S)

        draw_map(game, S, background, trucks, spawn_timer)
        S.clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()