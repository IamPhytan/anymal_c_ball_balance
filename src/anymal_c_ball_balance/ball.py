from pathlib import Path

import mujoco

BALL_RADIUS = 0.90
BALL_MASS = 15.0
BALL_FRICTION = (1.2, 0.01, 0.001)
BALL_HOLD_LEVELS = (1.0, 0.5, 0.2, 0.05, 0.0)

SOURCE_MESH_RADIUS = 0.1207
BALL_VISUAL_RADIUS = BALL_RADIUS + 0.002
BALL_MESH_SCALE = BALL_VISUAL_RADIUS / SOURCE_MESH_RADIUS

_ASSETS = Path(__file__).resolve().parent / "assets" / "basketball"


def get_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "ball"

    body = spec.worldbody.add_body(
        name="ball",
        pos=[0.0, 0.0, BALL_RADIUS],
    )

    # Texture
    tex = spec.add_texture(
        name="basketball_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(_ASSETS / "basketball.png"),
    )

    # Material
    mat = spec.add_material(name="basketball_mat")
    textures = list(mat.textures)
    textures[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "basketball_tex"
    mat.textures = textures
    mat.specular = 0.15
    mat.shininess = 0.15
    mat.reflectance = 0.0

    # Mesh
    mesh = spec.add_mesh(
        name="basketball_mesh",
        file=str(_ASSETS / "basketball.obj"),
        scale=[BALL_MESH_SCALE] * 3,
    )

    body.add_freejoint()

    # Geometry
    geom = body.add_geom(
        name="ball_sphere",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
    )
    geom.size = [BALL_RADIUS, 0.0, 0.0]
    geom.mass = BALL_MASS
    geom.rgba = [1.0, 1.0, 1.0, 0.0]  # invisible: the mesh below carries the look
    geom.group = 3
    geom.friction = list(BALL_FRICTION)

    geom.priority = 1  # the ball's friction wins over the floor's / the feet's
    geom.condim = 4  # torsional friction: a spinning ball under the feet is not free
    geom.contype = 1
    geom.conaffinity = 1

    v = body.add_geom(
        name="ball_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="basketball_mesh",
        material="basketball_mat",
        contype=0,
        conaffinity=0,
        mass=0.0,
        group=0,
    )

    v.contype = 0
    v.conaffinity = 0
    v.mass = 0.0
    v.group = 0

    body.add_site(
        name="ball_center",
        pos=[0.0, 0.0, 0.0],
        size=[0.004, 0.004, 0.004],
        group=3,
    )

    return spec


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity import Entity, EntityCfg

    ball = Entity(EntityCfg(spec_fn=get_ball_spec))
    viewer.launch(ball.spec.compile())
