import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


def make_vao(letter_vertices, letter_indices):
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # VBO
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, letter_vertices.nbytes,
                 letter_vertices, GL_STATIC_DRAW)

    # vertex
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

    # color
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    # indexing
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, letter_indices.nbytes,
                 letter_indices, GL_STATIC_DRAW)
    glBindVertexArray(0)

    return [vao, vbo, ebo]


def make_rotation(n, model_loc, translation, vao, letter_indices):
    # OBJECT TRANSFORMATION FOR L
    roty_letter = pyrr.matrix44.create_from_y_rotation(n * glfw.get_time())
    model_letter = pyrr.matrix44.multiply(scale, roty_letter)
    model_letter = pyrr.matrix44.multiply(model_letter, translation)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model_letter)

    # OBJECT ASSEMBLY AND RENDERING FOR LETTER
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, len(letter_indices), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

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

l_vertices = [
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

l_indices = [
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

l_vertices = np.array(l_vertices, dtype=np.float32)
l_indices = np.array(l_indices, dtype=np.uint32)

# C DEFINATION

c_vertices = [
    -0.1, 0.25, 0.0, 1.0, 0.0, 0.0,
    -0.1, 0.15, 0.0, 0.0, 1.0, 0.0,
    0.2, 0.15, 0.0, 0.0, 0.0, 1.0,
    0.2, 0.25, 0.0, 1.0, 1.0, 0.0,
    -0.2, 0.15, 0.0, 0.0, 1.0, 1.0,
    -0.2, -0.15, 0.0, 1.0, 0.0, 1.0,
    -0.1, -0.15, 0.0, 1.0, 0.0, 0.0,
    -0.1, -0.25, 0.0, 0.0, 1.0, 0.0,
    0.2, -0.25, 0.0, 0.0, 0.0, 1.0,
    0.2, -0.15, 0.0, 1.0, 1.0, 0.0,
    -0.1, 0.25, 0.1, 1.0, 0.0, 0.0,
    -0.1, 0.15, 0.1, 0.0, 1.0, 0.0,
    0.2, 0.15, 0.1, 0.0, 0.0, 1.0,
    0.2, 0.25, 0.1, 1.0, 1.0, 0.0,
    -0.2, 0.15, 0.1, 0.0, 1.0, 1.0,
    -0.2, -0.15, 0.1, 1.0, 0.0, 1.0,
    -0.1, -0.15, 0.1, 1.0, 0.0, 0.0,
    -0.1, -0.25, 0.1, 0.0, 1.0, 0.0,
    0.2, -0.25, 0.1, 0.0, 0.0, 1.0,
    0.2, -0.15, 0.1, 1.0, 1.0, 0.0,
]

c_indices = [
    0, 1, 3, 0, 2, 3,
    4, 5, 6, 4, 6, 1,
    6, 7, 8, 6, 8, 9,
    0, 4, 1,
    5, 7, 6,
    10, 11, 12, 10, 12, 13,
    14, 15, 16, 14, 16, 11,
    16, 17, 18, 16, 18, 19,
    10, 14, 11,
    15, 17, 16,
    1, 6, 16, 1, 16, 11,
    11, 1, 2, 11, 2, 12,
    16, 6, 9, 16, 9, 19,
    10, 0, 3, 10, 3, 13,
    17, 7, 8, 17, 8, 18,
    3, 2, 12, 3, 12, 13,
    9, 8, 18, 9, 18, 19,
    10, 14, 4, 10, 4, 0,
    15, 17, 7, 15, 7, 5,
    14, 15, 5, 14, 5, 4
]

c_vertices = np.array(c_vertices, dtype=np.float32)
c_indices = np.array(c_indices, dtype=np.uint32)

# A Defination

a_vertices = [
    -0.2, 0.15, 0.0, 1.0, 0.0, 0.0,
    -0.2, -0.25, 0.0, 0.0, 1.0, 0.0,
    -0.1, -0.25, 0.0, 0.0, 0.0, 1.0,
    -0.1, 0.15, 0.0, 1.0, 1.0, 0.0,
    0.2, 0.15, 0.0, 0.0, 1.0, 1.0,
    0.2, -0.25, 0.0, 1.0, 0.0, 1.0,
    0.1, -0.25, 0.0, 1.0, 0.0, 0.0,
    0.1, 0.15, 0.0, 0.0, 1.0, 0.0,
    -0.1, 0.05, 0.0, 0.0, 0.0, 1.0,
    -0.1, -0.05, 0.0, 1.0, 1.0, 0.0,
    0.1, -0.05, 0.0, 0.0, 1.0, 1.0,
    0.1, 0.05, 0.0, 1.0, 0.0, 1.0,
    -0.1, 0.25, 0.0, 1.0, 0.0, 0.0,
    0.1, 0.25, 0.0, 0.0, 1.0, 0.0,
    0.0, 0.15, 0.0, 0.0, 0.0, 1.0,
    -0.2, 0.15, 0.1, 1.0, 0.0, 0.0,
    -0.2, -0.25, 0.1, 0.0, 1.0, 0.0,
    -0.1, -0.25, 0.1, 0.0, 0.0, 1.0,
    -0.1, 0.15, 0.1, 1.0, 1.0, 0.0,
    0.2, 0.15, 0.1, 0.0, 1.0, 1.0,
    0.2, -0.25, 0.1, 1.0, 0.0, 1.0,
    0.1, -0.25, 0.1, 1.0, 0.0, 0.0,
    0.1, 0.15, 0.1, 0.0, 1.0, 0.0,
    -0.1, 0.05, 0.1, 0.0, 0.0, 1.0,
    -0.1, -0.05, 0.1, 1.0, 1.0, 0.0,
    0.1, -0.05, 0.1, 0.0, 1.0, 1.0,
    0.1, 0.05, 0.1, 1.0, 0.0, 1.0,
    -0.1, 0.25, 0.1, 1.0, 0.0, 0.0,
    0.1, 0.25, 0.1, 0.0, 1.0, 0.0,
    0.0, 0.15, 0.1, 0.0, 0.0, 1.0,
]

a_indices = [
    0, 1, 2, 0, 2, 3,
    4, 5, 6, 4, 6, 7,
    8, 9, 10, 8, 10, 11,
    12, 3, 7, 12, 7, 13,
    14, 3, 8,
    14, 11, 7,
    15, 16, 17, 15, 17, 18,
    19, 20, 21, 19, 21, 22,
    23, 24, 25, 23, 25, 26,
    27, 18, 22, 27, 22, 28,
    29, 18, 23,
    29, 22, 26,
    15, 16, 1, 15, 1, 0,
    19, 20, 5, 19, 5, 4,
    24, 17, 2, 24, 2, 9,
    25, 21, 6, 25, 6, 10,
    16, 1, 2, 16, 2, 17,
    21, 6, 5, 21, 5, 20,
    27, 15, 0, 27, 0, 12,
    12, 3, 0,
    27, 18, 15,
    13, 4, 19, 13, 19, 28,
    13, 7, 4,
    28, 22, 19,
    29, 23, 8, 29, 8, 14,
    29, 26, 11, 29, 11, 14,
    27, 12, 13, 27, 13, 28
]

a_vertices = np.array(a_vertices, dtype=np.float32)
a_indices = np.array(a_indices, dtype=np.uint32)

# SENDING DATA
vao, vbo, ebo = make_vao(l_vertices, l_indices)
vao1, vbo1, ebo1 = make_vao(c_vertices, c_indices)
vao2, vbo2, ebo2 = make_vao(a_vertices, a_indices)

# Transformation
scale = pyrr.Matrix44.from_scale(pyrr.Vector3([1, 1, 1]))

# Translation
translation = pyrr.Matrix44.from_translation([0.5, -0.5, 0])  # letter l
translation1 = pyrr.Matrix44.from_translation([0.5, 0.5, 0])  # Letter C
translation2 = pyrr.Matrix44.from_translation([-0.5, 0.5, 0])  # Letter A

# SHADER SPACE
shader = compileProgram(compileShader(
    vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)

model_loc = glGetUniformLocation(shader, "model")

# RENDERING SPACE
# SETUP THE COLOR FOR BACKGROUND
glClearColor(0.1, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)  # activate the z buffer
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # For letter L
    make_rotation(28, model_loc, translation, vao, l_indices)

    # For letter C
    make_rotation(30, model_loc, translation1, vao1, c_indices)

    # For letter A
    make_rotation(20, model_loc, translation2, vao2, a_indices)

    glfw.swap_buffers(window)

# MEMORY CLEARING
glDeleteBuffers(2, [vbo, ebo, vbo1, ebo1,])
glDeleteVertexArrays(1, [vao1, vao,])
glfw.terminate()
