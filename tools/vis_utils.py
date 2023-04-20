import pytorch3d

import torch
import imageio
import numpy as np
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import os

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): A square image to unproject (S, S, 3).
        mask (torch.Tensor): A binary mask for the image (S, S).
        depth (torch.Tensor): The depth map of the image (S, S).
        camera: The Pytorch3D camera to render the image.
    
    Returns:
        points (torch.Tensor): The 3D points of the unprojected image (N, 3).
        rgba (torch.Tensor): The rgba color values corresponding to the unprojected
            points (N, 4).
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates)
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    # For some reason, the Pytorch3D compositor does not apply a background color
    # unless the pointcloud is RGBA.
    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb


def load_cow_mesh(path="data/cow_mesh.obj"):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, _ = load_obj(path)
    faces = faces.verts_idx
    return vertices, faces

def render_mesh(predicted_mesh, save_path, args):
    color = torch.tensor([1.0, 0.0, 0.0], device=args.device)

    # obtain the textures
    vertex_src = predicted_mesh.verts_packed().reshape([len(predicted_mesh), -1, 3])  # (b, n_points, 3)
    textures = torch.ones(vertex_src.shape, device=args.device) * color

    textures = pytorch3d.renderer.TexturesVertex(textures)
    predicted_mesh.textures = textures

    # render
    render_360_mesh(predicted_mesh, save_path, dist=3.0)


def render_360_mesh(meshes, save_path, nof_angle=50, image_size=256, dist=2.0):
    '''
    * Input:
        meshes: a single match
    '''
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    device = get_device()
    meshes = meshes.to(device)
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)

    # Extend mesh 
    meshes = meshes.extend(nof_angle)

    # Get a batch of viewing angles.
    elev = torch.ones(nof_angle) * 30
    azim = torch.linspace(-180, 180, nof_angle)

    # All the cameras helper methods support mixed type inputs and broadcasting. So we can
    # view the camera from the same distance and specify dist=2.7 as a float,
    # and then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim)

    # generate cameras
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    # the cow is facing -z direction, so use the point light to light up its face
    lights = pytorch3d.renderer.PointLights(
        location=[[0, 0, -4.0]], device=device)

    rend = renderer(meshes, cameras=cameras, lights=lights)
    rend = rend[:, ..., :3].detach().cpu().numpy()

    # save the images into a gif
    torus_images = [(rend[i]*255).astype(np.uint8)
                    for i in range(rend.shape[0])]
    imageio.mimsave(save_path, torus_images, fps=25)

def render_360_point_cloud(point_cloud, save_path, dist=3.0, image_size = 256, nof_angle=50, device=None):
    if device is None:
        device = get_device()
    
    # obtain point cloud renderer
    points_renderer = get_points_renderer(image_size=image_size)

    # Get a batch of viewing angles.
    point_cloud = point_cloud.extend(nof_angle).to(device)
    elev = torch.ones(nof_angle)
    azim = torch.linspace(-180, 180, nof_angle)

    # create multiple R and T by looking at different viewpoints
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(
        dist=dist, elev=elev, azim=azim)

    # since the pytorch3d internal uses Point= point@R+t instead of using Point=R @ point+t,
    # we need to add R.t() to compensate that.
    # generate cameras
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R, T=T, device=device)
    print("Begin to render Point Clouds...")
    rend = points_renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[:, ..., :3]
    print("Point Cloud Render complete")

    # save the images into a gif
    plant_images = [(rend[i]*255).astype(np.uint8)
                    for i in range(rend.shape[0])]
    imageio.mimsave(save_path, plant_images, fps=25)