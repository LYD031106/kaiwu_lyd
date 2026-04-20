#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Path finding utilities for Robot Vacuum.
清扫大作战寻路工具函数。
"""

from collections import deque

import numpy as np


DIRS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS_8 = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _to_tuple(pos):
    """Convert position to int tuple (x, z).

    将坐标转换为整数二元组 (x, z)。
    """
    if pos is None or len(pos) != 2:
        raise ValueError("position must be a 2-element sequence")
    return int(pos[0]), int(pos[1])


def _in_bounds(x, z, rows, cols):
    """Check whether (x, z) is inside map bounds.

    判断坐标是否在地图范围内。
    """
    return 0 <= x < rows and 0 <= z < cols


def _build_blocked_mask(status_map, passable_status=None):
    """Build blocked mask from status map.

    根据 status 地图构建障碍掩码。

    Args:
        status_map: 2D ndarray，整张地图的 status。
        passable_status: 可通行 status 集合；若为 None，则默认仅 0 不可通行。
    """
    if passable_status is None:
        return np.asarray(status_map) == 0

    passable_set = set(passable_status)
    return ~np.isin(status_map, list(passable_set))


def reconstruct_path(parent, end_pos):
    """Reconstruct path from parent map.

    根据父节点表还原完整路径。
    """
    path = []
    cur = end_pos
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def shortest_path_to_status(
    cur_pos,
    status_map,
    target_status,
    passable_status=None,
    allow_diagonal=False,
):
    """Find shortest path from current position to nearest target status.

    从当前位置出发，找到最近的目标 status，并返回最短路径。

    Args:
        cur_pos: 当前位置 `(x, z)`。
        status_map: 2D 地图，每个格子的值表示其 status。
        target_status: 目标格子的 status 值。
        passable_status: 可通行的 status 集合。默认仅 `0` 视为障碍，其他均可通行。
        allow_diagonal: 是否允许斜向移动。默认 False，仅四方向。

    Returns:
        dict:
            - found: 是否找到目标
            - target_pos: 最近目标位置；未找到时为 None
            - distance: 最短路径步数；未找到时为 -1
            - path: 从起点到目标的路径（含起点和终点）；未找到时为空列表
    """
    status_map = np.asarray(status_map)
    if status_map.ndim != 2:
        raise ValueError("status_map must be a 2D array")

    rows, cols = status_map.shape
    start = _to_tuple(cur_pos)
    if not _in_bounds(start[0], start[1], rows, cols):
        raise ValueError("cur_pos is out of map bounds")

    blocked = _build_blocked_mask(status_map, passable_status=passable_status)
    if blocked[start[0], start[1]]:
        return {
            "found": False,
            "target_pos": None,
            "distance": -1,
            "path": [],
        }

    if status_map[start[0], start[1]] == target_status:
        return {
            "found": True,
            "target_pos": start,
            "distance": 0,
            "path": [start],
        }

    dirs = DIRS_8 if allow_diagonal else DIRS_4
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    distance = {start: 0}

    while queue:
        x, z = queue.popleft()
        for dx, dz in dirs:
            nx, nz = x + dx, z + dz
            nxt = (nx, nz)
            if not _in_bounds(nx, nz, rows, cols):
                continue
            if nxt in visited or blocked[nx, nz]:
                continue

            visited.add(nxt)
            parent[nxt] = (x, z)
            distance[nxt] = distance[(x, z)] + 1

            if status_map[nx, nz] == target_status:
                return {
                    "found": True,
                    "target_pos": nxt,
                    "distance": distance[nxt],
                    "path": reconstruct_path(parent, nxt),
                }

            queue.append(nxt)

    return {
        "found": False,
        "target_pos": None,
        "distance": -1,
        "path": [],
    }


def shortest_distance_to_status(
    cur_pos,
    status_map,
    target_status,
    passable_status=None,
    allow_diagonal=False,
):
    """Find shortest distance from current position to nearest target status.

    返回到最近目标 status 的最短距离，不关心完整路径时可直接调用。
    """
    result = shortest_path_to_status(
        cur_pos=cur_pos,
        status_map=status_map,
        target_status=target_status,
        passable_status=passable_status,
        allow_diagonal=allow_diagonal,
    )
    return result["distance"]
