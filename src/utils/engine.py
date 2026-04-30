import os
import sys
sys.path.append(os.getcwd())
import env

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np
import cv2

import math
import shutil

def draw_center_cube():
    glBegin(GL_QUADS)

    # +X face (Red)
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f( 1, -1, -1)
    glVertex3f( 1,  1, -1)
    glVertex3f( 1,  1,  1)
    glVertex3f( 1, -1,  1)

    # -X face (Green)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-1, -1,  1)
    glVertex3f(-1,  1,  1)
    glVertex3f(-1,  1, -1)
    glVertex3f(-1, -1, -1)

    # +Y face (Blue)
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1,  1, -1)
    glVertex3f(-1,  1,  1)
    glVertex3f( 1,  1,  1)
    glVertex3f( 1,  1, -1)

    # -Y face (Yellow)
    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(-1, -1,  1)
    glVertex3f(-1, -1, -1)
    glVertex3f( 1, -1, -1)
    glVertex3f( 1, -1,  1)

    # +Z face (Magenta)
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(-1, -1,  1)
    glVertex3f( 1, -1,  1)
    glVertex3f( 1,  1,  1)
    glVertex3f(-1,  1,  1)

    # -Z face (Cyan)
    glColor3f(0.0, 1.0, 1.0)
    glVertex3f( 1, -1, -1)
    glVertex3f(-1, -1, -1)
    glVertex3f(-1,  1, -1)
    glVertex3f( 1,  1, -1)

    glEnd()

def draw_skybox():
    glPushMatrix()

    size = 100.0
    nQuads = 50 
    dy = (2 * size) / nQuads

    def get_color(y):
        y_mapped = ((y + size) / (2 * size)) * 255.0
        t = 1.0 / (1.0 + math.exp(-0.1 * (y_mapped - 130)))
        r = t * 178
        g = t * 204
        b = t * 255
        return (r / 255.0, g / 255.0, b / 255.0)

    glDepthMask(GL_FALSE)

    # Draw the +X face (x = +size)
    glBegin(GL_QUADS)
    for i in range(nQuads):
        y0 = -size + i * dy
        y1 = y0 + dy
        color0 = get_color(y0)
        color1 = get_color(y1)
        glColor3f(*color0)
        glVertex3f(size, y0, -size)
        glColor3f(*color1)
        glVertex3f(size, y1, -size)
        glColor3f(*color1)
        glVertex3f(size, y1, size)
        glColor3f(*color0)
        glVertex3f(size, y0, size)
    glEnd()

    # Draw the -X face (x = -size)
    glBegin(GL_QUADS)
    for i in range(nQuads):
        y0 = -size + i * dy
        y1 = y0 + dy
        color0 = get_color(y0)
        color1 = get_color(y1)
        glColor3f(*color0)
        glVertex3f(-size, y0, size)
        glColor3f(*color1)
        glVertex3f(-size, y1, size)
        glColor3f(*color1)
        glVertex3f(-size, y1, -size)
        glColor3f(*color0)
        glVertex3f(-size, y0, -size)
    glEnd()

    # Draw the +Z face (z = +size)
    glBegin(GL_QUADS)
    for i in range(nQuads):
        y0 = -size + i * dy
        y1 = y0 + dy
        color0 = get_color(y0)
        color1 = get_color(y1)
        glColor3f(*color0)
        glVertex3f(-size, y0, size)
        glColor3f(*color1)
        glVertex3f(-size, y1, size)
        glColor3f(*color1)
        glVertex3f(size, y1, size)
        glColor3f(*color0)
        glVertex3f(size, y0, size)
    glEnd()

    # Draw the -Z face (z = -size)
    glBegin(GL_QUADS)
    for i in range(nQuads):
        y0 = -size + i * dy
        y1 = y0 + dy
        color0 = get_color(y0)
        color1 = get_color(y1)
        glColor3f(*color0)
        glVertex3f(size, y0, -size)
        glColor3f(*color1)
        glVertex3f(size, y1, -size)
        glColor3f(*color1)
        glVertex3f(-size, y1, -size)
        glColor3f(*color0)
        glVertex3f(-size, y0, -size)
    glEnd()

    glDepthMask(GL_TRUE)

    glPopMatrix()

def draw_checkerboard(rows=8, cols=8, square_size=1.0, z=0.0):
    half_width = (cols * square_size) / 2.0
    half_height = (rows * square_size) / 2.0
    border_thickness = square_size

    # Draw the white border as four separate quads
    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_QUADS)
    
    # Top border (above the board)
    glVertex3f(-half_width - border_thickness,  half_height, z)
    glVertex3f( half_width + border_thickness,  half_height, z)
    glVertex3f( half_width + border_thickness,  half_height + border_thickness, z)
    glVertex3f(-half_width - border_thickness,  half_height + border_thickness, z)
    
    # Bottom border (below the board)
    glVertex3f(-half_width - border_thickness, -half_height - border_thickness, z)
    glVertex3f( half_width + border_thickness, -half_height - border_thickness, z)
    glVertex3f( half_width + border_thickness, -half_height, z)
    glVertex3f(-half_width - border_thickness, -half_height, z)
    
    # Left border (to the left of the board)
    glVertex3f(-half_width - border_thickness, -half_height, z)
    glVertex3f(-half_width,               -half_height, z)
    glVertex3f(-half_width,                half_height, z)
    glVertex3f(-half_width - border_thickness, half_height, z)
    
    # Right border (to the right of the board)
    glVertex3f( half_width,               -half_height, z)
    glVertex3f( half_width + border_thickness, -half_height, z)
    glVertex3f( half_width + border_thickness,  half_height, z)
    glVertex3f( half_width,                half_height, z)
    glEnd()
    
    # Draw the inner checkerboard squares
    glBegin(GL_QUADS)
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                glColor3f(1.0, 1.0, 1.0)
            else:
                glColor3f(0.0, 0.0, 0.0)
            
            x0 = col * square_size - half_width
            x1 = (col + 1) * square_size - half_width
            y0 = row * square_size - half_height
            y1 = (row + 1) * square_size - half_height
            
            glVertex3f(x0, y0, z)
            glVertex3f(x1, y0, z)
            glVertex3f(x1, y1, z)
            glVertex3f(x0, y1, z)
    glEnd()

def init_gl(display):
    glViewport(0, 0, display[0], display[1])
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 150.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_MULTISAMPLE)

def apply_radial_distortion(img, k1, k2):
    """
    Applies radial distortion to an image.
    The distortion model is:
      x_dist = x*(1 + k1*r^2 + k2*r^4)
    where (x,y) are normalized coordinates.
    """
    height, width = img.shape[:2]
    # Compute approximate focal length in pixels from 45° FOV:
    fov = 45.0 * math.pi/180.0
    f = (width/2.0) / math.tan(fov/2.0)
    cx = width / 2.0
    cy = height / 2.0

    # Create the mapping arrays
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)
    
    # For each pixel, compute the distorted location
    for y in range(height):
        for x in range(width):
            # normalized coordinates (relative to center)
            xn = (x - cx) / f
            yn = (y - cy) / f
            r2 = xn*xn + yn*yn
            factor = 1 + k1*r2 + k2*(r2**2)
            x_dist = xn * factor
            y_dist = yn * factor
            # map back to pixel coordinates
            map_x[y, x] = x_dist * f + cx
            map_y[y, x] = y_dist * f + cy

    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return distorted

def capture_viewport(display, filename="screenshot.png", k1=-0.5, k2=0.4):
    width, height = display
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = pygame.image.fromstring(data, (width, height), "RGB")
    image = pygame.transform.flip(image, False, True)
    # Convert the Pygame surface to a numpy array.
    np_image = pygame.surfarray.array3d(image)
    np_image = np.transpose(np_image, (1, 0, 2))  # shape: (height, width, 3)
    
    # Apply radial distortion with the given k1 and k2.
    distorted_np_image = apply_radial_distortion(np_image, k1, k2)
    
    # Convert back to Pygame surface and save.
    distorted_image = pygame.surfarray.make_surface(np.transpose(distorted_np_image, (1, 0, 2)))

    pygame.image.save(distorted_image, filename)
    print(f"Screenshot saved to {filename}")

def load_texture(image_path):
    """Load an image file as an OpenGL texture."""
    try:
        surface = pygame.image.load(image_path)
    except Exception as e:
        print(f"Unable to load texture {image_path}: {e}")
        return None
    image_data = pygame.image.tostring(surface, "RGBA", 1)
    width, height = surface.get_rect().size
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    return texture_id

def load_mtl(filename):
    """
    Loads a Wavefront .mtl file.
    """
    materials = {}
    current_material = None
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'newmtl':
                current_material = parts[1]
                materials[current_material] = {}
            elif current_material is None:
                continue
            elif parts[0] in ['Ka', 'Kd', 'Ks']:
                # store as a tuple or list
                materials[current_material][parts[0]] = list(map(float, parts[1:]))
            elif parts[0] == 'Ns':
                materials[current_material]['Ns'] = float(parts[1])
            elif parts[0] in ['d', 'Tr']:
                # handle transparency
                materials[current_material]['d'] = float(parts[1])
            elif parts[0] == 'illum':
                materials[current_material]['illum'] = int(parts[1])
            elif parts[0] == 'map_Kd':
                texture_filename = parts[1]
                dir_name = os.path.dirname(filename)
                texture_path = os.path.join(dir_name, texture_filename)
                texture_id = load_texture(texture_path)
                if texture_id is not None:
                    materials[current_material]['map_Kd'] = texture_id
    return materials

def load_obj(filename, mtl_filename=None):
    """
    Loads a Wavefront .obj file and optionally its .mtl file if provided or if
    the .obj has an 'mtllib' line inside.
    """
    vertices = []
    texcoords = []
    normals = []
    faces = []
    current_material = None
    
    # Start with no materials; we’ll load them if we see mtllib or if mtl_filename is given
    materials = {}

    # If user explicitly passes a .mtl file, load it right away
    if mtl_filename is not None:
        materials = load_mtl(mtl_filename)

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            
            if parts[0] == 'v':
                vertices.append(list(map(float, parts[1:4])))
            elif parts[0] == 'vt':
                texcoords.append(list(map(float, parts[1:3])))
            elif parts[0] == 'vn':
                normals.append(list(map(float, parts[1:4])))

            elif parts[0] == 'mtllib':
                # If the .obj references a .mtl file, load it (only if we haven't already)
                if mtl_filename is None:
                    dir_name = os.path.dirname(filename)
                    mtl_path = os.path.join(dir_name, parts[1])
                    materials = load_mtl(mtl_path)

            elif parts[0] == 'usemtl':
                current_material = parts[1]
            elif parts[0] == 'f':
                face = []
                for v in parts[1:]:
                    vals = v.split('/')
                    v_idx  = int(vals[0]) - 1
                    vt_idx = int(vals[1]) - 1 if len(vals) >= 2 and vals[1] != '' else None
                    vn_idx = int(vals[2]) - 1 if len(vals) >= 3 and vals[2] != '' else None
                    face.append((v_idx, vt_idx, vn_idx))
                faces.append((face, current_material))

    return vertices, texcoords, normals, faces, materials

def create_gl_list_from_obj(obj_filename: str, mtl_filename: str=None):
    """
    Loads an .obj (and optionally its .mtl) file and compiles it into an OpenGL display list.
    """
    vertices, texcoords, normals, faces, materials = load_obj(obj_filename, mtl_filename)
    
    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)

    for face, mat in faces:
        if mat and mat in materials:
            mat_data = materials[mat]

            # 1) If there's a texture in map_Kd, enable texturing
            if 'map_Kd' in mat_data:
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, mat_data['map_Kd'])
                # Set color to white so we see the true texture color
                glColor3f(1.0, 1.0, 1.0)
            else:
                glDisable(GL_TEXTURE_2D)

            # 2) Optionally set a material color from Kd
            if 'Kd' in mat_data:
                # If you use glColorMaterial, you can just do:
                glColor3fv(mat_data['Kd'])

                # Or if you want advanced lighting control:
                # glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, mat_data['Kd'] + [1.0])
                
        else:
            # No material -> just disable texturing
            glDisable(GL_TEXTURE_2D)
            glColor3f(1.0, 1.0, 1.0)

        # Pick the primitive
        if len(face) == 3:
            glBegin(GL_TRIANGLES)
        elif len(face) == 4:
            glBegin(GL_QUADS)
        else:
            glBegin(GL_POLYGON)

        for (v_idx, vt_idx, vn_idx) in face:
            if vn_idx is not None:
                glNormal3fv(normals[vn_idx])
            if vt_idx is not None:
                glTexCoord2fv(texcoords[vt_idx])
            glVertex3fv(vertices[v_idx])
        glEnd()

    glEndList()
    return list_id

def get_distorted_chessboard(path):
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)

    info = pygame.display.Info()
    scale = 0.80
    aspect_ratio = info.current_w / info.current_h
    display = (int(info.current_w * scale), int(info.current_h * scale))
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    init_gl(display)
    glTranslatef(0.0, 0.0, -15)

    # Reset scene and draw static objects
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_skybox()

    # Draw checkerboard
    glPushMatrix()
    glRotatef(5, 2, 1, 1)
    draw_checkerboard(rows=10, cols=int(10*aspect_ratio))
    glPopMatrix()

    capture_viewport(display, path)

    pygame.display.flip()
    pygame.time.wait(10)

    pygame.quit()

def get_chessboard(path):
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)

    info = pygame.display.Info()
    scale = 0.80
    aspect_ratio = info.current_w / info.current_h
    display = (int(info.current_w * scale), int(info.current_h * scale))
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    init_gl(display)
    glTranslatef(0.0, 0.0, -15)

    # Reset scene and draw static objects
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_skybox()

    # Draw checkerboard
    glPushMatrix()
    draw_checkerboard(rows=11, cols=int(11*aspect_ratio))
    glPopMatrix()

    capture_viewport(display, path, k1=0, k2=0)

    pygame.display.flip()
    pygame.time.wait(10)

    pygame.quit()

def get_object_images(obj, texture, path, views=20, angle=10):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    pygame.init()

    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)

    info = pygame.display.Info()
    scale = 0.80
    aspect_ratio = info.current_w / info.current_h
    display = (int(info.current_w * scale), int(info.current_h * scale))
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    init_gl(display)
    glTranslatef(0.0, -1.5, -10)

    obj_display_list = create_gl_list_from_obj(obj, mtl_filename=texture)
    
    for i in range(views):
        # Reset scene and draw static objects
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_skybox()

        glPushMatrix()
        glTranslatef(0.0, 0.0, -2.0)
        glRotatef(15, 1, 0, 0)
        glRotatef(angle, 0, 1, 0)
        glTranslatef(-1.5, 0.0, 1.0)
        glCallList(obj_display_list)
        glPopMatrix()

        capture_viewport(display, path / f'object_{i}.png', k1=0.0, k2=0.0)

        pygame.display.flip()
        angle += 20

    pygame.quit()


def get_test_obj_images(path1, path2):
    pygame.init()

    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)

    info = pygame.display.Info()
    scale = 0.80
    aspect_ratio = info.current_w / info.current_h
    display = (int(info.current_w * scale), int(info.current_h * scale))
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    init_gl(display)
    glTranslatef(0.0, -1.5, -10)

    obj_display_list = create_gl_list_from_obj(env.p5.test_obj, mtl_filename=env.p5.test_texture)

    # Reset scene and draw static objects
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_skybox()

    glPushMatrix()
    glTranslatef(0.0, 0.0, -2.0)
    glRotatef(15, 1, 0, 0)
    glRotatef(25, 0, 1, 0)
    glTranslatef(-1.5, 0.0, 1.0)
    glCallList(obj_display_list)
    glPopMatrix()

    capture_viewport(display, path1, k1=0.0, k2=0.0)

    pygame.display.flip()
    pygame.time.wait(10)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    draw_skybox()

    glPushMatrix()
    glTranslatef(0.0, 0.0, -2.0)
    glRotatef(15, 1, 0, 0)
    glRotatef(35, 0, 1, 0)
    glTranslatef(-1.5, 0.0, 1.0)
    glCallList(obj_display_list)
    glPopMatrix()

    capture_viewport(display, path2, k1=0.0, k2=0.0)

    pygame.display.flip()
    pygame.time.wait(10)

    pygame.quit()

def draw_3d_points(points3D):
    """
    Draws the provided 3D points as GL_POINTS in the current OpenGL context.
    points3D can be a list of (x, y, z) or Nx3 NumPy array.
    """
    glPointSize(3.0)           # make points a bit bigger
    glColor3f(1.0, 0.0, 0.0)   # set color to red (change as desired)

    glBegin(GL_POINTS)
    for p in points3D:
        x, y, z = p
        glVertex3f(x, y, z)
    glEnd()

def show_points(points3D):
    pygame.init()

    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)

    info = pygame.display.Info()
    scale = 0.80
    aspect_ratio = info.current_w / info.current_h
    display = (int(info.current_w * scale), int(info.current_h * scale))
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)

    init_gl(display)
    glTranslatef(0.0, -2.0, -15)

    clock = pygame.time.Clock()
    running = True


    pos = (0., 0., 0.)
    translation_speed = 5.0
    while running:
        dt = clock.tick(60) / 1000.0

        # Parse Inputs
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                display = event.size
                screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
                init_gl(display)

                # Process continuous key presses
        keys = pygame.key.get_pressed()
        x, y, z = pos
        if keys[pygame.K_LEFT]:
            x -= translation_speed * dt
        if keys[pygame.K_RIGHT]:
            x += translation_speed * dt
        if keys[pygame.K_UP]:
            y += translation_speed * dt
        if keys[pygame.K_DOWN]:
            y -= translation_speed * dt
        if keys[pygame.K_w]:
            z += translation_speed * dt
        if keys[pygame.K_s]:
            z -= translation_speed * dt
        pos = (x, y, z)

        # Reset scene and draw static objects
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_skybox()

        # Draw rotating points
        glPushMatrix()
        glTranslatef(*pos)
        glRotatef(15, 1, 0, 0)
        # glRotatef(angle, 0, 1, 0)
        # glTranslatef(-1.5, 0.0, 1.0)
        draw_3d_points(points3D)
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()
    sys.exit()

def main():
    pygame.init()

    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 16)

    info = pygame.display.Info()
    scale = 0.80
    aspect_ratio = info.current_w / info.current_h
    display = (int(info.current_w * scale), int(info.current_h * scale))
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)

    init_gl(display)
    glTranslatef(0.0, -2.0, -15)

    obj_display_list = create_gl_list_from_obj(env.p5.test_obj, mtl_filename=env.p5.test_texture)

    clock = pygame.time.Clock()
    running = True

    angle = 0

    while running:
        angle += 1
        dt = clock.tick(60) / 1000.0

        # Parse Inputs
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                display = event.size
                screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
                init_gl(display)

        # Reset scene and draw static objects
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_skybox()

        # Draw checkerboard
        glPushMatrix()
        # glRotatef(5, 2, 1, 1)
        # draw_checkerboard(rows=10, cols=int(10*aspect_ratio))
        glPopMatrix()

        # Draw rotating cube
        glPushMatrix()
        glTranslatef(0.0, 0.0, -2.0)
        glRotatef(15, 1, 0, 0)
        glRotatef(angle, 0, 1, 0)
        glTranslatef(-1.5, 0.0, 1.0)
        glCallList(obj_display_list)
        glPopMatrix()

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
