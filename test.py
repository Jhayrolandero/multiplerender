import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr

# SHADERS GLSL

vertex_src = """
# version 330

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;

out vec3 v_color;

void main()
{
    gl_Position = model * vec4(position, 1.0);
    v_color = color;
}
"""

fragment_src = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 1.0);
}
"""

# window init
glfw.init()
window = glfw.create_window(800, 600, "Act window", None, None)

if not window:
    glfw.terminate()
    exit()

glfw.make_context_current(window)

# OBJECT CREATION
# CUBE_VERTEX DEFINITION

# cube_vertices = [

#     -0.5, 0.5, -0.5, 1.0, 0.0, 0.0,  # 1
#     -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,  # 2
#     0.5, -0.5, -0.5, 0.0, 0.0, 1.0,  # 3
#     0.5, 0.5, -0.5, 1.0, 0.0, 1.0,  # 4
#     -0.5, 0.5, 0.5, 1.0, 1.0, 0.0,  # 5
#     -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,  # 6
#     0.5, -0.5, 0.5, 0.0, 1.0, 0.0,  # 7
#     0.5, 0.5, 0.5, 1.0, 0.0, 0.0]   # 8

cube_vertices = [
    -0.2, 0.25, -0.1, 1.0, 0.0, 0.0,
    -0.2, -0.15, -0.1, 0.0, 1.0, 0.0,
    -0.2, -0.25, -0.1, 0.0, 0.0, 1.0,
    0.2, -0.25, -0.1, 1.0, 1.0, 0.0,
    0.2, -0.15, -0.1, 0.0, 1.0, 1.0,
    -0.1, -0.15, -0.1, 1.0, 0.0, 1.0,
    -0.1, 0.25, -0.1, 1.0, 0.0, 0.0,
    - 0.2, 0.25, 0.0, 1.0, 0.0, 0.0,
    -0.2, -0.15, 0.0, 0.0, 1.0, 0.0,
    -0.2, -0.25, 0.0, 0.0, 0.0, 1.0,
    0.2, -0.25, 0.0, 1.0, 1.0, 0.0,
    0.2, -0.15, 0.0, 0.0, 1.0, 1.0,
    -0.1, -0.15, 0.0, 1.0, 0.0, 1.0,
    -0.1, 0.25, 0.0, 1.0, 0.0, 0.0
]

# TRIANGLE CREATTION BY INDEX METHOD
# cube_indices = [0, 1, 2, 0, 2, 3,
#                 3, 2, 6, 3, 6, 7,
#                 7, 6, 5, 7, 5, 4,
#                 4, 5, 1, 4, 1, 0,
#                 4, 0, 3, 4, 3, 7,
#                 1, 5, 6, 1, 6, 2]

cube_indices = [
    0, 1, 5, 0, 5, 6,
    6, 5, 12, 6, 12, 13,
    13, 12, 8, 13, 8, 7,
    7, 8, 1, 7, 1, 0,
    1, 0, 6, 1, 6, 13,
    1, 2, 3, 1, 3, 4,
    4, 3, 10, 4, 10, 11,
    11, 10, 9, 11, 9, 8,
    8, 9, 2, 8, 2, 1,
    2, 9, 10, 2, 10, 3
]

cube_vertices = np.array(cube_vertices, dtype=np.float32)
cube_indices = np.array(cube_indices, dtype=np.uint32)

# PYRAMID DEFINATION

pyramid_vertices = [
    0.0, 0.5, 0.0, 1.0, 0.0, 0.0,
    -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,
    0.5, -0.5, -0.5, 0.0, 0.0, 1.0,
    0.5, -0.5, 0.5, 1.0, 1.0, 1.0,
    -0.5, -0.5, 0.5, 0.0, 1.0, 1.0,
]

pyramid_indices = [
    0, 1, 2,
    0, 2, 3,
    0, 3, 4,
    0, 4, 1
]

pyramid_vertices = np.array(pyramid_vertices, dtype=np.float32)
pyramid_indices = np.array(pyramid_indices, dtype=np.uint32)

# SENDING DATA
# ================================================================
#                             CUBE

# vao
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# VBO
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes,
             cube_vertices, GL_STATIC_DRAW)

# vertex
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

# color
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

# indexing
ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes,
             cube_indices, GL_STATIC_DRAW)
glBindVertexArray(0)

# =============================================================
#                           pyramid

# vao
vao1 = glGenVertexArrays(1)
glBindVertexArray(vao1)

# VBO
vbo1 = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo1)
glBufferData(GL_ARRAY_BUFFER, pyramid_vertices.nbytes,
             pyramid_vertices, GL_STATIC_DRAW)

# vertex
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

# color
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

# indexing
ebo1 = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo1)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, pyramid_indices.nbytes,
             pyramid_indices, GL_STATIC_DRAW)
glBindVertexArray(0)
# =============================================================


# Transformation
scale = pyrr.Matrix44.from_scale(pyrr.Vector3([1, 1, 1]))

# Translation
translation = pyrr.Matrix44.from_translation([-0.5, 0, 0])
translation1 = pyrr.Matrix44.from_translation([0.5, 0, 0])

# SHADER SPACE
shader = compileProgram(compileShader(
    vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)

model_loc = glGetUniformLocation(shader, "model")
# view_loc = glGetUniformLocation(shader, "view")
# glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

# RENDERING SPACE
# SETUP THE COLOR FOR BACKGROUND
glClearColor(0.1, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)  # activate the z buffer
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # OBJECT TRANSFORMATION FOR CUBE
    roty_cube = pyrr.matrix44.create_from_y_rotation(5 * glfw.get_time())
    model_cube = pyrr.matrix44.multiply(scale, roty_cube)
    model_cube = pyrr.matrix44.multiply(model_cube, translation)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_cube)

    # OBJECT ASSEMBLY AND RENDERING FOR CUBE
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(cube_indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    # OBJECT TRANSFORMATION FOR PYRAMID
    roty_pyramid = pyrr.matrix44.create_from_y_rotation(-5 * glfw.get_time())
    model_pyramid = pyrr.matrix44.multiply(scale, roty_pyramid)
    model_pyramid = pyrr.matrix44.multiply(model_pyramid, translation1)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_pyramid)

    # OBJECT ASSEMBLY AND RENDERING FOR PYRAMID
    glBindVertexArray(vao1)
    glDrawElements(GL_TRIANGLES, len(pyramid_indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    glfw.swap_buffers(window)
# MEMORY CLEARING
glDeleteBuffers(2, [vbo, ebo, vbo1, ebo1,])
glDeleteVertexArrays(1, [vao1, vao,])
glfw.terminate()
