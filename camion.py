import time
import sys
import random

import numpy as np
import pygame
import math
from numba import njit


WALKABLE_INIT = 100
BLOCKED_INIT = 999
SHOW_DIST_DEBUG = False
BLOCK_CELL_DENSITY = 8
MOVE_MODE = "continuous"  # "discrete" or "continuous"
SPAWN_RATE_PER_SPAWNER = 1.0
CUSTOMER_SPEED_MIN = 1.5
CUSTOMER_SPEED_MAX = 2.5
SHOPPING_LIST_MIN = 2
SHOPPING_LIST_MAX = 6
PICK_TIME_SECONDS = 0.8
PAY_TIME_SECONDS = 1.0
TOTAL_CUSTOMERS_TARGET = 1

SYTADIN_COLORS = [
    (183, 228, 199),
    (183, 228, 199),
    (183, 228, 199),
    (144, 214, 180),
    (104, 200, 160),
    (178, 222, 122),
    (222, 230, 109),
    (255, 214, 102),
    (255, 183, 77),
    (255, 138, 51),
    (240, 84, 44),
    (214, 40, 40),
]


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

                min_neighbor = BLOCKED_INIT

                if x + 1 < w and out_dist[x + 1, y] < min_neighbor:
                    min_neighbor = out_dist[x + 1, y]
                if x - 1 >= 0 and out_dist[x - 1, y] < min_neighbor:
                    min_neighbor = out_dist[x - 1, y]
                if y + 1 < h and out_dist[x, y + 1] < min_neighbor:
                    min_neighbor = out_dist[x, y + 1]
                if y - 1 >= 0 and out_dist[x, y - 1] < min_neighbor:
                    min_neighbor = out_dist[x, y - 1]

                new_val = min_neighbor + 1
                if new_val < out_dist[x, y]:
                    out_dist[x, y] = new_val
                    changed = True

    return iterations


##########################################################################
#
#  Color
#
##########################################################################


class Color:
    pink   = (255, 105, 180)
    orange = (255, 165, 0)
    cyan   = (0, 255, 255)
    red    = (255, 0, 0)
    yellow = (255, 255, 0)
    blue   = (0, 0, 255)
    white  = (255, 255, 255)
    black  = (0, 0, 0)
    gray   = (128,128,128)



##########################################################################
#
#  Etat du jeu
#
##########################################################################


class GameData:
    # index 0 = rester sur place

    def __init__(self, map_text):
        self.map, self.mapW, self.mapH = self._mapCreate(map_text)

        self.stands_by_type = self._findStandsByType()
        self.stands = self._flattenStands(self.stands_by_type)

        # Track occupancy for each stand cell: None or customer_id
        self.stand_occupancy = {s: None for s in self.stands}
        self.spawns = self._getSpawns()

        self.checkouts = self._getCheckouts()

        self.stands_np = self._pointsToNumpy(self.stands)
        self.checkouts_np = self._pointsToNumpy(self.checkouts)

        self.base_dist = self._buildBaseDistance()
        self.dist = self.base_dist.copy()
        self.density = np.zeros((self.mapW, self.mapH), dtype=np.int32)
        self.checkout_capacity = max(1, len(self.checkouts))
        self.checkout_waiting = []
        self.checkout_paying = []

    def _promoteCheckoutQueue(self):
        while self.checkout_waiting and len(self.checkout_paying) < self.checkout_capacity:
            self.checkout_paying.append(self.checkout_waiting.pop(0))

    def enqueueCheckout(self, customer_id):
        if customer_id in self.checkout_waiting or customer_id in self.checkout_paying:
            return

        self.checkout_waiting.append(customer_id)
        self._promoteCheckoutQueue()

    def isCustomerPaying(self, customer_id):
        return customer_id in self.checkout_paying

    def isCustomerWaiting(self, customer_id):
        return customer_id in self.checkout_waiting

    def finishCheckout(self, customer_id):
        if customer_id in self.checkout_paying:
            self.checkout_paying.remove(customer_id)
        elif customer_id in self.checkout_waiting:
            self.checkout_waiting.remove(customer_id)

        self._promoteCheckoutQueue()

    def getCheckoutWaitingCount(self):
        return len(self.checkout_waiting)

    # Stand occupancy helpers
    def isStandOccupied(self, stand_cell):
        return self.stand_occupancy.get(stand_cell) is not None

    def occupyStand(self, stand_cell, customer_id):
        # Reserve stand if free. Return True when reserved, False if already occupied.
        if stand_cell not in self.stand_occupancy:
            return False
        if self.stand_occupancy[stand_cell] is None:
            self.stand_occupancy[stand_cell] = customer_id
            return True
        return False

    def releaseStand(self, stand_cell, customer_id=None):
        # Release stand only if it is currently held by given customer_id, or unconditionally when customer_id is None
        if stand_cell not in self.stand_occupancy:
            return
        if customer_id is None or self.stand_occupancy[stand_cell] == customer_id:
            self.stand_occupancy[stand_cell] = None


    ######################################################################
    # map
    ######################################################################

    def _mapCreate(self, text):
        rows = text.strip().splitlines()
        rows = [line.strip() for line in rows]
        rows = [line.replace(".", "") for line in rows]

        for line in rows:
            if len(line) != len(rows[0]):
                print("different length :", line)
                raise Exception("Map length error")

        rows.reverse()  # met l'origine en bas a gauche
        grid = [[ c for c in row] for row in rows]

        array = np.array(grid, dtype='U1').transpose()
        width, height = array.shape
        return array, width, height



    def _findStandsByType(self):
        stands_by_type = {}

        for x in range(self.mapW):
            for y in range(self.mapH):
                cell = self.map[x, y]

                # Any rack symbol can be a shopping target.
                if cell in (' ', 'M', 'W', 'S'):
                    continue

                if cell not in stands_by_type:
                    stands_by_type[cell] = []
                stands_by_type[cell].append((x, y))

        return stands_by_type

    def _flattenStands(self, stands_by_type):
        stands = []
        for points in stands_by_type.values():
            stands.extend(points)
        return stands

    def _getSpawns(self):
        spawns = []

        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'W':
                    spawns.append((x, y))

        return spawns

    def _getCheckouts(self):
        checkouts = []

        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] == 'S':
                    checkouts.append((x, y))

        return checkouts

    def _pointsToNumpy(self, points):
        if not points:
            return np.empty((0, 2), dtype=np.int32)
        return np.array(points, dtype=np.int32)

    def _buildBaseDistance(self):
        base = np.full((self.mapW, self.mapH), BLOCKED_INIT, dtype=np.int32)
        for x in range(self.mapW):
            for y in range(self.mapH):
                if self.map[x, y] in (' ', 'W'):
                    base[x, y] = WALKABLE_INIT
        return base

    ######################################################################
    # Calcul de la carte des distances (Etape 1)
    ######################################################################

    def computeDistanceMap(self, targets_xy, out_dist):
        if targets_xy is None or targets_xy.shape[0] == 0:
            out_dist[:, :] = self.base_dist
            return 0

        max_iterations = self.mapW * self.mapH
        return compute_distance_map_numba(
            self.base_dist,
            targets_xy,
            max_iterations,
            out_dist,
        )


#######################################
#
#  Screen control
#
##########################################################################

DEFAULT_ZOOM = 40
MIN_ZOOM = 12
ZOOM = DEFAULT_ZOOM


class Screen:
    def __init__(self, nx, ny, title="ESIEE - Supermarket Simulator"):
        pygame.display.init()
        pygame.font.init()

        global ZOOM
        info = pygame.display.Info()

        # Keep margins so the game window stays visible on screen.
        max_w = max(320, info.current_w - 80)
        max_h = max(240, info.current_h - 120)
        zoom_w = max_w // nx
        zoom_h = max_h // (ny + 1)
        ZOOM = max(MIN_ZOOM, min(DEFAULT_ZOOM, zoom_w, zoom_h))

        self.SCREEN_WIDTH  = nx * ZOOM
        self.SCREEN_HEIGHT = (ny + 1) * ZOOM
        self.EPAISS = max(2, ZOOM // 5)

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        self.font22 = pygame.font.SysFont("Arial", max(14, int(ZOOM * 0.55)), bold=True)
        self.font_debug = pygame.font.SysFont("Arial", max(10, int(ZOOM * 0.35)))

    ######################################################################
    # conversions coordonnees
    ######################################################################

    def screen_to_grid(self, xpix, ypix):
        xpix += ZOOM // 2
        gx = xpix // ZOOM - 1
        ypix -= ZOOM // 2
        gy = (self.SCREEN_HEIGHT - ypix) // ZOOM - 2
        return gx, gy

    def grid_to_screen(self, x, y):
        sx = (x ) * ZOOM
        sy = self.SCREEN_HEIGHT - (y + 1) * ZOOM
        return sx, sy

    ######################################################################
    # primitives de dessin
    ######################################################################

    def clear(self,coul=Color.black):
        self.screen.fill(coul)

    def drawRect(self,x,y,L,H,coul,width = 0):
        x1, y1 = self.grid_to_screen(x, y)
        HH = ZOOM * H
        LL = ZOOM * L
        pygame.draw.rect(self.screen, coul, (x1, y1-HH, LL, HH),width = width)


    def drawCircle(self, x, y, r, color = Color.red):
        R = r*ZOOM
        cx, cy = self.grid_to_screen(x , y )
        pygame.draw.circle(self.screen, color, (cx, cy) , R)


    def drawText(self, x, y, txt, color=Color.white, bigfont=False, centered=False):
        font = self.font22 if bigfont else self.font_debug
        xx, yy = self.grid_to_screen(x, y)
        text_surface = font.render(str(txt), True, color)
        height = text_surface.get_height()
        width  = text_surface.get_width()

        if (centered):
            self.screen.blit(text_surface, (xx-width//2, yy - height//2))
        else:
            self.screen.blit(text_surface, (xx, yy - height))

    def drawTriangle(self, A,B,C, color=Color.orange):
        p1 = self.grid_to_screen(*A)
        p2 = self.grid_to_screen(*B)
        p3 = self.grid_to_screen(*C)

        pygame.draw.polygon(self.screen,color,
            [
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (int(p3[0]), int(p3[1])),
            ],
        )

    def debugDist(self, dist_map):
        for x in range(G.mapW):
            for y in range(G.mapH):
                if G.map[x,y] == ' ':
                    v = dist_map[x,y]
                    self.drawText(x+0.5,y+0.5,str(v),Color.white,False,True)


    def show(self):
        pygame.display.flip()





##########################################################################
#
#  PROJET
#
##########################################################################


T = """
.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M.M
.M.M.M.M.M.M.M.M.M.M.M.M.M.U. .U. .U. .U. . .O.O.O.O.O.O.O.M
.M.M.M.M.M.M.M.M.M.M.M.M.M.U. .U. .U. .U. . . . . . . . . .M
.M.B.B.B.B.B. .A.A.A.A.A.A.A. . . . . . . . .O.O.O.M.O.O. .M
.M.B. . . . . . . . . . . . . . . . . . . . . . . . . . . .M
.M.B. .B.B.B. .A.A.A.A.A.A.A. .U. .U. .J. . .O.O.O.O.O.O. .M
.M.B. .B.B.B. .A.M.M.M.M.M.A. .U. .U. .J. . .J.J.J.J.J.J. .M
.M.L. . . . . . . . . . . . . . . . . . . . . . . . . . . .M
.M.L. .L.L.L.L. .F. .F. .T.T. .T.T. .P.P. . .J.J.J.M.J.J. .M
.M.L. . . . . . .F. .F. .T.T. .T.T. .P.P. . .J.J.J.J.J.J. .M
.M.L. .L.L.M.L. .F. .F. .M.T. .T.T. .M.P. . . . . . . . . .M
.M.L. . . . . . .F. .F. .T.T. .T.T. .P.P. . .J.J.J.J.J.J. .M
.M.L. .L.L.L.L. .F. .F. .T.T. .T.T. .P.P. . .J.J.J.J.J.J. .M
.M.L. . . . . . . . . . . . . . . . . . . . . . . . . . . .M
.M.L. . . . . . . . . . . . . . . . . . . . . . . . . . .R.M
.M.G. .G.G. .G.E. .E.E. .E.E. .E.E. .P. . . .R.R. .M.R. .R.M
.M.G. . . . .G.E. .E.E. .E.E. .E.E. .P. . . . . . . . . .R.M
.M.G. .G.G. .G.E. .E.E. .M.E. .E.E. .P. . . .R.R. .R.R. .R.M
.M.G. .G.G. .G.E. .E.E. .E.E. .E.E. .P. . . . . . . . . .R.M
.M.G. .G.G. .G.E. .E.E. .E.E. .E.E. .P. . . .R.R. .R.R. .R.M
.M.G. . . . . . . . . . . . . . . . . . . . . . . . . . .R.M
.M.M.M.M.M.S.M.M.S.M.M.S.M.M.S.M.M.S.M.W.W.W.M.M.M.M.M.M.M.M
"""

TableCoul = {
'G' : (132, 211, 245), #surGelés
'M' : (133, 152, 167), #mur
'B' : (235, 52, 36)  , #boucherie
'L' : (146, 186, 61) , #légumes
'E' : (173, 115, 93) , #Epicerie
'F' : (159, 117, 195), #fruits
'A' : (172, 36, 74)  , #Alcool/Vin
'T' : (255, 210, 227), #textile
'P' : (254, 255, 55) , #promotions
'J' : (252, 147, 30) , #jouets
'R' : (255, 223, 148), #électRonique
'S' : (220, 220, 220), #Sorties
'O' : (14, 147, 255) , #bOissons
'U' : (171, 45, 144) , #sUcrée
'W' : (40,40,40)          #spawn
}


class Customer:
    def __init__(self, game, spawn_cell):
        global NEXT_CUSTOMER_ID

        self.customer_id = NEXT_CUSTOMER_ID
        NEXT_CUSTOMER_ID += 1

        sx, sy = spawn_cell

        # Etape 8: random x in [0,1] within the spawn cell.
        self.x = sx + random.random()
        self.y = sy + 0.5
        self.shopping_list = self._buildShoppingList(game)
        self.targets = []
        self.dir       = (0.0, 1.0)
        self.base_speed = CUSTOMER_SPEED_MIN + random.random() * (CUSTOMER_SPEED_MAX - CUSTOMER_SPEED_MIN)
        self.move_budget = 0.0
        self.state     = "shopping"
        self.shopping_done = False
        self.pick_time_left = 0.0
        self.pay_time_left = 10.0
        self.active_stand_cell = None
        self.dist      = np.empty_like(game.base_dist)

        self._updateShoppingTargets(game)

        if not self.shopping_list:
            self.shopping_done = True
            self._startCheckout(game)

        self._refreshDistanceMap(game)

    def _buildShoppingList(self, game):
        product_types = list(game.stands_by_type.keys())
        if not product_types:
            return []

        requested = random.randint(SHOPPING_LIST_MIN, SHOPPING_LIST_MAX)
        return [random.choice(product_types) for _ in range(requested)]

    def _updateShoppingTargets(self, game):
        if not self.shopping_list:
            self.targets = []
            return

        needed_types = set(self.shopping_list)
        targets = []
        for product_type in needed_types:
            stands = list(game.stands_by_type.get(product_type, []))
            # Prefer unoccupied stands. If none free, include all so customer can wait.
            free = [s for s in stands if not game.isStandOccupied(s)]
            if free:
                targets.extend(free)
            else:
                targets.extend(stands)
        self.targets = targets

    def _pickAdjacentNeededStand(self, game):
        if not self.shopping_list:
            return None

        needed_types = set(self.shopping_list)
        cx, cy = self._gridPos()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < game.mapW and 0 <= ny < game.mapH):
                continue
            cell_type = game.map[nx, ny]
            if cell_type in needed_types:
                # Only pick if stand is free (capacity 1). If occupied, skip.
                if not game.isStandOccupied((nx, ny)):
                    return cell_type, (nx, ny)

        return None

    def _refreshDistanceMap(self, game):
        if self.state == "shopping" and self.targets:
            targets_xy = np.array(self.targets, dtype=np.int32)
            game.computeDistanceMap(targets_xy, self.dist)
        elif self.state == "to_checkout" and game.checkouts_np.shape[0] > 0:
            game.computeDistanceMap(game.checkouts_np, self.dist)
        else:
            self.dist[:, :] = game.base_dist

    def _gridPos(self):
        return int(self.x), int(self.y)

    def _cellCenter(self, x, y):
        return x + 0.5, y + 0.5

    def _isWalkable(self, game, x, y):
        if not (0 <= x < game.mapW and 0 <= y < game.mapH):
            return False
        return game.map[x, y] in (' ', 'W')

    def _adjacentTargetIndex(self, target_list):
        cx, cy = self._gridPos()
        for i, (tx, ty) in enumerate(target_list):
            if abs(tx - cx) + abs(ty - cy) == 1:
                return i
        return None

    def _bestNeighborCell(self, game, cx, cy, current_dist, density_map):
        best_dist = current_dist
        candidates = []

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = cx + dx, cy + dy
            if not self._isWalkable(game, nx, ny):
                continue

            # During checkout approach, allow denser cells to avoid deadlock.
            max_allowed_density = BLOCK_CELL_DENSITY
            if self.state == "to_checkout":
                max_allowed_density = BLOCK_CELL_DENSITY + 3

            if density_map[nx, ny] >= max_allowed_density:
                continue

            nd = self.dist[nx, ny]
            if nd < best_dist:
                best_dist = nd
                candidates = [(nx, ny)]
            elif nd == best_dist:
                candidates.append((nx, ny))

        if not candidates:
            return None

        min_local_density = min(int(density_map[nx, ny]) for nx, ny in candidates)
        least_crowded = [
            (nx, ny)
            for nx, ny in candidates
            if int(density_map[nx, ny]) == min_local_density
        ]

        if len(least_crowded) == 1:
            return least_crowded[0]

        dx, dy = self.dir
        norm = math.hypot(dx, dy)
        if norm > 1e-8:
            dx /= norm
            dy /= norm

            best_score = -2.0
            best_candidates = []
            for nx, ny in least_crowded:
                vx = nx - cx
                vy = ny - cy
                vnorm = math.hypot(vx, vy)
                if vnorm < 1e-8:
                    score = 1.0
                else:
                    score = (vx / vnorm) * dx + (vy / vnorm) * dy

                if score > best_score + 1e-8:
                    best_score = score
                    best_candidates = [(nx, ny)]
                elif abs(score - best_score) <= 1e-8:
                    best_candidates.append((nx, ny))

            if best_candidates:
                return random.choice(best_candidates)

        return random.choice(least_crowded)

    def _densitySpeedPenalty(self, density_value):
        # Softer slowdown: no penalty until crowd reaches 4, then linear.
        crowd_over = max(0, int(density_value) - 4)
        return 1.0 + 0.20 * crowd_over

    def _startCheckout(self, game):
        # Checkout/queue must only happen after shopping is complete.
        if not self.shopping_done:
            return
        # Release any stand reservation when leaving shopping.
        if self.active_stand_cell is not None:
            game.releaseStand(self.active_stand_cell, self.customer_id)
        self.active_stand_cell = None
        if game.checkouts_np.shape[0] > 0:
            self.state = "to_checkout"
        else:
            self.state = "done"
        self.dir = (0, 0)

    def _startPaying(self):
        # active_stand_cell should already be released when moving to checkout,
        # but clear to be safe.
        self.active_stand_cell = None
        self.state = "paying"
        self.pay_time_left = PAY_TIME_SECONDS
        self.dir = (0, 0)

    def _startPicking(self, product_type, stand_cell):
        # Try to reserve the stand. If already occupied, abort picking.
        reserved = G.occupyStand(stand_cell, self.customer_id)
        if not reserved:
            return

        self.state = "picking"
        self.pick_time_left = PICK_TIME_SECONDS
        self.active_stand_cell = stand_cell
        self.picked_product_type = product_type
        self.dir = (0.0, 0.0)

    def _finishPicking(self, game):
        if self.picked_product_type in self.shopping_list:
            self.shopping_list.remove(self.picked_product_type)

        self.picked_product_type = None
        # Release stand reservation
        if self.active_stand_cell is not None:
            game.releaseStand(self.active_stand_cell, self.customer_id)
        self.active_stand_cell = None
        self._updateShoppingTargets(game)

        if self.shopping_list:
            self.state = "shopping"
            self._refreshDistanceMap(game)
        else:
            self.shopping_done = True
            self._startCheckout(game)

    def update(self, game, dt, density_map):
        if self.state == "done":
            return

        if self.state == "paying":
            self.pay_time_left -= dt
            if self.pay_time_left <= 0:
                game.finishCheckout(self.customer_id)
                self.state = "done"
            return

        if self.state == "queueing":
            if game.isCustomerPaying(self.customer_id):
                self._startPaying()
            else:
                if not game.isCustomerWaiting(self.customer_id):
                    game.enqueueCheckout(self.customer_id)
                self.dir = (0.0, 0.0)
            return

        if self.state == "picking":
            self.pick_time_left -= dt
            if self.pick_time_left <= 0:
                self._finishPicking(game)
            return

        # Chaque client recalcule sa propre carte des distances.
        self._refreshDistanceMap(game)
        if MOVE_MODE == "discrete":
            self.moveDiscrete(game, dt, density_map)
        else:
            self.moveContinuous(game, dt, density_map)

    def _handleArrival(self, game, active_targets):
        if self.state == "shopping":
            picked = self._pickAdjacentNeededStand(game)
            if picked is not None:
                product_type, stand_cell = picked
                self._startPicking(product_type, stand_cell)
        elif self.state == "to_checkout":
            if not self.shopping_done:
                self.state = "shopping"
                self._refreshDistanceMap(game)
                return

            game.enqueueCheckout(self.customer_id)
            if game.isCustomerPaying(self.customer_id):
                self._startPaying()
            else:
                self.state = "queueing"

        self.dir = (0.0, 0.0)

    def moveDiscrete(self, game, dt, density_map):
        if self.state == "shopping":
            active_targets = self.targets
        elif self.state == "to_checkout":
            active_targets = game.checkouts
        else:
            active_targets = []

        if not active_targets:
            if self.state == "shopping":
                self._startCheckout(game)
            return

        cx, cy = self._gridPos()
        current_dist = self.dist[cx, cy]

        if current_dist <= 1:
            self._handleArrival(game, active_targets)
            return

        # Density-based speed kept in discrete mode using a movement budget.
        local_density = int(density_map[cx, cy])
        speed_penalty = self._densitySpeedPenalty(local_density)
        corrected_speed = self.base_speed / speed_penalty
        self.move_budget += corrected_speed * dt

        if self.move_budget < 1.0:
            self.dir = (0.0, 0.0)
            return

        best = self._bestNeighborCell(game, cx, cy, current_dist, density_map)
        if best is None:
            self.dir = (0.0, 0.0)
            return

        nx, ny = best
        self.x, self.y = self._cellCenter(nx, ny)
        self.dir = (float(nx - cx), float(ny - cy))
        self.move_budget -= 1.0

    def moveContinuous(self, game, dt, density_map):
        if self.state == "shopping":
            active_targets = self.targets
        elif self.state == "to_checkout":
            active_targets = game.checkouts
        else:
            active_targets = []

        if not active_targets:
            if self.state == "shopping":
                self._startCheckout(game)
            return

        cx, cy = self._gridPos()
        current_dist = self.dist[cx, cy]

        # Le target est sur un stand (non traversable). On s'arrete sur une
        # case d'allee adjacente (distance 1).
        if current_dist <= 1:
            self._handleArrival(game, active_targets)
            return

        best = self._bestNeighborCell(game, cx, cy, current_dist, density_map)

        if best is None:
            self.dir = (0.0, 0.0)
            return

        nx, ny = best
        tx, ty = self._cellCenter(nx, ny)

        vx = tx - self.x
        vy = ty - self.y
        dist_to_center = math.hypot(vx, vy)
        if dist_to_center < 1e-8:
            self.x, self.y = tx, ty
            self.dir = (0.0, 0.0)
            return

        self.dir = (vx / dist_to_center, vy / dist_to_center)

        # v_corrigee = v / max(1, nb_clients_sur_case - 2)
        local_density = int(density_map[cx, cy])
        speed_penalty = self._densitySpeedPenalty(local_density)
        corrected_speed = self.base_speed / speed_penalty
        step = corrected_speed * dt

        if step >= dist_to_center:
            self.x, self.y = tx, ty
        else:
            self.x += self.dir[0] * step
            self.y += self.dir[1] * step

    def drawCustomer(self):
        if self.state == "done":
            return

        x,y = self.x, self.y
        dx, dy = self.dir
        norm = math.hypot(dx, dy)
        if ( norm > 0.001) :
            dx = dx / norm
            dy = dy / norm
            lx,ly = -dy,dx # rot90
            l = 0.5  # longueur de la fleche
            s = 0.15  # largeur
            A = x + l * dx , y + l * dy
            B = x + s * lx , y + s * ly
            C = x - s * lx , y - s * ly
            S.drawTriangle(A,B,C,Color.black)
        else:
            # Keep stationary customers visible (e.g. paying at checkout).
            S.drawCircle(x, y, 0.12, Color.black)

    def drawTargets(self):
        return


class Spawner:
    def __init__(self, x, y, spawn_rate=2.0, max_clients=1):
        self.x = x
        self.y = y
        self.spawn_rate = float(spawn_rate)
        self.max_clients = int(max_clients)
        self.spawned_count = 0
        self.time_acc = 0.0

    def update(self, dt, game, customers, max_new_clients=None):
        if max_new_clients is not None and max_new_clients <= 0:
            return 0

        if self.spawned_count >= self.max_clients:
            return 0

        self.time_acc += dt * self.spawn_rate
        to_spawn = int(self.time_acc)
        if to_spawn <= 0:
            return 0

        self.time_acc -= to_spawn

        remaining = self.max_clients - self.spawned_count
        to_spawn = min(to_spawn, remaining)

        if max_new_clients is not None:
            to_spawn = min(to_spawn, max_new_clients)

        if to_spawn <= 0:
            return 0

        for _ in range(to_spawn):
            customers.append(Customer(game, (self.x, self.y)))

        self.spawned_count += to_spawn
        return to_spawn


HEADLESS = ('--headless' in sys.argv)

G = GameData(T)
S = None if HEADLESS else Screen(G.mapW, G.mapH)
NEXT_CUSTOMER_ID = 1
CUSTOMERS = []
SPAWNERS = [Spawner(x, y, spawn_rate=SPAWN_RATE_PER_SPAWNER, max_clients=TOTAL_CUSTOMERS_TARGET) for x, y in G.spawns]
TOTAL_SPAWNED = 0
GAME_FINISHED = False


def buildBackgroundBuffer():
    buffer_surface = pygame.Surface((S.SCREEN_WIDTH, S.SCREEN_HEIGHT))
    buffer_surface.fill(Color.black)

    for x in range(G.mapW):
        for y in range(G.mapH):
            cell_id = G.map[x, y]
            if cell_id == ' ':
                continue

            color = TableCoul[cell_id]
            x1, y1 = S.grid_to_screen(x, y)
            hh = ZOOM
            ll = ZOOM
            pygame.draw.rect(buffer_surface, color, (x1, y1 - hh, ll, hh), width=0)
            pygame.draw.rect(buffer_surface, Color.black, (x1, y1 - hh, ll, hh), width=2)

    return buffer_surface


BACKGROUND = buildBackgroundBuffer() if not HEADLESS else None


def updateDensityMap():
    G.density[:, :] = 0

    for cust in CUSTOMERS:
        if cust.state == "done":
            continue

        cx, cy = cust._gridPos()
        if 0 <= cx < G.mapW and 0 <= cy < G.mapH and G.map[cx, cy] in (' ', 'W'):
            G.density[cx, cy] += 1


def drawSytadinOverlay():
    max_idx = len(SYTADIN_COLORS) - 1

    for x in range(G.mapW):
        for y in range(G.mapH):
            if G.map[x, y] not in (' ', 'W'):
                continue

            density_level = int(G.density[x, y])
            color = SYTADIN_COLORS[min(density_level, max_idx)]
            S.drawRect(x, y, 1, 1, color)
            S.drawRect(x, y, 1, 1, Color.black, 1)


def drawActiveStandDots():
    for cust in CUSTOMERS:
        if cust.state != "picking" or cust.active_stand_cell is None:
            continue

        sx, sy = cust.active_stand_cell
        S.drawCircle(sx + 0.5, sy + 0.5, 0.12, Color.black)


def drawDensityNumbers():
    stand_counts = {}

    for cust in CUSTOMERS:
        if cust.state != "picking" or cust.active_stand_cell is None:
            continue

        key = cust.active_stand_cell
        stand_counts[key] = stand_counts.get(key, 0) + 1

    for (sx, sy), count in stand_counts.items():
        S.drawText(sx + 0.5, sy + 0.5, str(count), color=Color.black, bigfont=True, centered=True)





def drawMap():
    S.screen.blit(BACKGROUND, (0, 0))
    drawSytadinOverlay()
    drawDensityNumbers()


    for cust in CUSTOMERS:
        cust.drawCustomer()

    done_count = sum(1 for c in CUSTOMERS if c.state == "done")
    shopping_count = sum(1 for c in CUSTOMERS if c.state == "shopping")
    picking_count = sum(1 for c in CUSTOMERS if c.state == "picking")
    to_exit_count = sum(1 for c in CUSTOMERS if c.state == "to_checkout")
    queueing_count = sum(1 for c in CUSTOMERS if c.state == "queueing")
    paying_count = sum(1 for c in CUSTOMERS if c.state == "paying")
    active_count = TOTAL_SPAWNED - done_count
    waiting_count = G.getCheckoutWaitingCount()


    S.drawText(0,-1, "  SPACE = pause", color=Color.white, bigfont=True)
    S.drawText(9,-1, f"spawned={TOTAL_SPAWNED}/{TOTAL_CUSTOMERS_TARGET} active={active_count} shop={shopping_count} pick={picking_count} to_exit={to_exit_count} queue={queueing_count} wait={waiting_count} pay={paying_count} done={done_count}", color=Color.white, bigfont=True)

    if GAME_FINISHED:
        S.drawText(G.mapW * 0.5, G.mapH * 0.5, "Simulation terminee", color=Color.white, bigfont=True, centered=True)

    if SHOW_DIST_DEBUG and CUSTOMERS:
        S.debugDist(CUSTOMERS[0].dist)

    S.show()


def playOneTurn(dt):
    global TOTAL_SPAWNED, GAME_FINISHED

    if PAUSE_FLAG or GAME_FINISHED:
        return

    remaining_to_spawn = max(0, TOTAL_CUSTOMERS_TARGET - TOTAL_SPAWNED)
    if remaining_to_spawn > 0:
        for spawner in SPAWNERS:
            if remaining_to_spawn <= 0:
                break

            new_clients = spawner.update(dt, G, CUSTOMERS, max_new_clients=remaining_to_spawn)
            TOTAL_SPAWNED += new_clients
            remaining_to_spawn -= new_clients

    updateDensityMap()

    for cust in CUSTOMERS:
        cust.update(G, dt, G.density)

    updateDensityMap()

    done_count = sum(1 for c in CUSTOMERS if c.state == "done")
    if TOTAL_SPAWNED >= TOTAL_CUSTOMERS_TARGET and done_count >= TOTAL_CUSTOMERS_TARGET:
        GAME_FINISHED = True


def run_headless(timeout_seconds=600):
    # Run the simulation without rendering; print progress once per second.
    global GAME_FINISHED
    start = time.time()
    logic_fps = 30 if MOVE_MODE != 'discrete' else 7
    step = 1.0 / logic_fps
    last_print = 0

    while not GAME_FINISHED:
        playOneTurn(step)
        now = time.time()
        if now - last_print >= 1.0:
            done_count = sum(1 for c in CUSTOMERS if c.state == "done")
            active_count = TOTAL_SPAWNED - done_count
            waiting_count = G.getCheckoutWaitingCount()
            print(f"t={int(now-start)}s spawned={TOTAL_SPAWNED}/{TOTAL_CUSTOMERS_TARGET} active={active_count} done={done_count} waiting={waiting_count}")
            last_print = now

        if now - start > timeout_seconds:
            print("Headless run timeout reached")
            break

    print("Headless run finished:", "GAME_FINISHED=" , GAME_FINISHED)


##########################################################################
#
#  Boucle principale
#
##########################################################################

PAUSE_FLAG = False

if HEADLESS:
    run_headless()
else:
    LOGIC_CALL = pygame.USEREVENT + 1
    LOGIC_fps = 7 if MOVE_MODE == "discrete" else 30
    pygame.time.set_timer(LOGIC_CALL, int(1000 / LOGIC_fps))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                PAUSE_FLAG = not PAUSE_FLAG
            elif event.type == LOGIC_CALL:
                playOneTurn(1 / LOGIC_fps)

        if GAME_FINISHED:
            running = False

        drawMap()
        S.clock.tick(60)

    pygame.quit()
