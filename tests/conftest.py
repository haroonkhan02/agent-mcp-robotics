"""Synthetic fixtures: a tiny RST corpus and a small fixed world. No network, no models."""

from __future__ import annotations

import pytest

from robomcp.docs import DocsIndex, sections_from
from robomcp.rst import load_rst
from robomcp.world import Robot, World

TOPICS_RST = """\
.. redirect-from::

    Old/Path

.. _Topics:

Topics
======

.. contents:: Table of Contents
   :local:

Topics are one of the three primary styles of interfaces provided by ROS 2.

.. image:: images/topic.gif

Durability
----------

* Transient local: the publisher becomes responsible for persisting samples for "late-joining"
  subscriptions.

.. note:: See the :doc:`QoS guide <../About-QoS>` for ``ros2 topic echo`` details.

Commands
--------

.. code-block:: console

   $ ros2 topic list
"""

PARAMS_RST = """\
Parameters
==========

A parameter is a configuration value of a node, for example ``use_sim_time``.

Setting a parameter
-------------------

Use ``ros2 param set`` to change a parameter at run time.
"""


@pytest.fixture
def rst_root(tmp_path):
    root = tmp_path / "source"
    (root / "Concepts").mkdir(parents=True)
    (root / "Concepts" / "About-Topics.rst").write_text(TOPICS_RST)
    (root / "Concepts" / "About-Parameters.rst").write_text(PARAMS_RST)
    (root / "Releases").mkdir()
    (root / "Releases" / "Jazzy-Complete-Changelog.rst").write_text(
        "Changelog\n=========\n\ntopic parameter topic parameter\n")
    return root


@pytest.fixture
def docs_index(rst_root):
    docs = [load_rst(p, rst_root) for p in sorted(rst_root.rglob("About-*.rst"))]
    return DocsIndex(sections_from(docs))


PARAMS = {"max_speed": 1.0, "inflation_radius": 0.35, "goal_tolerance": 0.25}


@pytest.fixture
def world():
    """amr_1 idle at shelf_A, amr_2 charging at dock_1, amr_3 with a lidar fault at shelf_D."""
    robots = [
        Robot(id="amr_1", x=6.0, y=2.0, at="shelf_A", battery=80.0, params=dict(PARAMS)),
        Robot(id="amr_2", x=1.0, y=1.0, at="dock_1", battery=55.0, charging=True,
              params=dict(PARAMS)),
        Robot(id="amr_3", x=12.0, y=2.0, at="shelf_D", battery=90.0, fault="LIDAR_TIMEOUT",
              params=dict(PARAMS)),
    ]
    shelves = {"shelf_A": ["gripper", "wheel_kit"], "shelf_B": ["imu_board"], "shelf_C": [],
               "shelf_D": ["lidar_unit"], "shelf_E": []}
    w = World(robots, shelves)
    w._diag(robots[2], "ERROR", "lidar_driver", "LIDAR_TIMEOUT: node stopped responding")
    return w
