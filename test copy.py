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
    gl_Position = vec4(position, 1.0);
    v_color = color;
}
"""

# vertex_src = """
# # version 330

# layout(location = 0) in vec3 position;
# layout(location = 1) in vec3 color;

# uniform mat4 model;
# uniform mat4 view;
# uniform mat4 proj;

# out vec3 v_color;

# void main()
# {
#     gl_Position = view * vec4(position, 1.0);
#     v_color = color;
# }
# """

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
# VERTEX DEFINITION

cube_vertices = [-0.5, 0.5, -0.5, 1.0, 0.0, 0.0,  # 1
                 -0.5, -0.5, -0.5, 0.0, 1.0, 0.0,  # 2
                 0.5, -0.5, -0.5, 0.0, 0.0, 1.0,  # 3
                 0.5, 0.5, -0.5, 1.0, 0.0, 1.0,  # 4
                 -0.5, 0.5, 0.5, 1.0, 1.0, 0.0,  # 5
                 -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,  # 6
                 0.5, -0.5, 0.5, 0.0, 1.0, 0.0,  # 7
                 0.5, 0.5, 0.5, 1.0, 0.0, 0.0]   # 8

# TRIANGLE CREATTION BY INDEX METHOD
cube_indices = [0, 1, 2, 0, 2, 3,
                3, 2, 6, 3, 6, 7,
                7, 6, 5, 7, 5, 4,
                4, 5, 1, 4, 1, 0,
                4, 0, 3, 4, 3, 7,
                1, 5, 6, 1, 6, 2]

cube_vertices = np.array(cube_vertices, dtype=np.float32)
cube_indices = np.array(cube_indices, dtype=np.uint32)
# SENDING DATA
# vao
vao = glGenVertexArrays(1)
glBindVertexArray(vao)

# VBO
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData()

# vertex
glEnableVertexAttribArray(0)
glVertexAttribPointer()

# color
glEnableVertexAttribArray(1)
glVertexAttribPointer()

# indexing
ebo = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
glBufferData()
glBindVertexArray(0)


# = ================================
# vao1 = glGenVertexArrays(1)
# glBindVertexArray(vao1)

# # vbo
# vbo = glGenBuffers(1)
# glBindBuffer(GL_ARRAY_BUFFER, vbo)
# glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# # ebo
# ebo = glGenBuffers(1)
# glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
# glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

# # position
# glEnableVertexAttribArray(0)
# glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

# # color
# glEnableVertexAttribArray(1)
# glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
# TRANSFORMATION

# VIEW_MATRIX
# cam_pos = np.array([1, 0, 0])
# cam_tar = np.array([0, 0, 0])
# g_up = np.array([0, 1, 0])

# cam_for = pyrr.vector.normalise(cam_tar - cam_pos)
# print(cam_for)
# cam_right = pyrr.vector.normalise(pyrr.vector3.cross(g_up, cam_for))
# cam_up = pyrr.vector.normalise(pyrr.vector3.cross(cam_for, cam_right))
# cam_for = -cam_for

# rotation = np.resize([np.append(cam_right, 0),
#                       np.append(cam_up, 0),
#                       np.append(cam_for, 0),
#                       [0, 0, 0, 1]], [4, 4])

# translation = [[1, 0, 0, -np.dot(cam_right, cam_pos)],
#                [0, 1, 0, -np.dot(cam_up, cam_pos)],
#                [0, 0, 1, -np.dot(cam_for, cam_pos)],
#                [0, 0, 0, 1]]

# view = pyrr.matrix44.multiply(translation, rotation).T


# PROJECTION_MATRIX

# SHADER SPACE
shader = compileProgram(compileShader(
    vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
glUseProgram(shader)

# model_loc = glGetUniformLocation(shader, "model")
# view_loc = glGetUniformLocation(shader, "view")
# glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

# RENDERING SPACE
# SETUP THE COLOR FOR BACKGROUND
glClearColor(0.1, 0.1, 0.1, 1)
glEnable(GL_DEPTH_TEST)  # activate the z buffer
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # OBJECT TRANSFORMATION

    # OBJECT ASSEMBLY AND RENDERING
    glBindVertexArray(vao1)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

# MEMORY CLEARING
glDeleteBuffers(2, [vbo, ebo,])
glDeleteVertexArrays(1, [vao1,])
glfw.terminate()
