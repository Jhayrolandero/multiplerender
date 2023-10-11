import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


class Object:
    
    def __init__(self, vertices, indices, translation, shader):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.translation = translation
        self.shader = shader
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.vao, self.vbo, self.ebo = self.make_vao()
        self.scale = pyrr.Matrix44.from_scale(pyrr.Vector3([1, 1, 1]))

    def make_vao(self):
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

        return vao, vbo, ebo

    def make_rotation(self, n):
        roty = pyrr.matrix44.create_from_y_rotation(n * glfw.get_time())
        model = pyrr.matrix44.multiply(self.scale, roty)
        model = pyrr.matrix44.multiply(model, self.translation)
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)
        glBindVertexArray(0)




class Renderer:
    def __init__(self, width, height, title):
        glfw.init()
        self.window = glfw.create_window(width, height, title, None, None)

        if not self.window:
            glfw.terminate()
            exit()

        glfw.make_context_current(self.window)

        glClearColor(0.1, 0.1, 0.1, 1)
        glEnable(GL_DEPTH_TEST)

        self.objects = []
        self.n = []

    def add_object(self, obj):
        self.objects.append(obj)
        
    def object_rotation(self, n):
        self.n.append(n)

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            for obj, n in zip(self.objects, self.n):
                obj.make_rotation(n) 

            glfw.swap_buffers(self.window)

    def cleanup(self):
        for obj in self.objects:
            glDeleteBuffers(2, [obj.vbo, obj.ebo])
            glDeleteVertexArrays(1, [obj.vao])

        glfw.terminate()


VERTEX_SRC = """
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

FRAGMENT_SRC = """
# version 330

in vec3 v_color;

out vec4 out_color;

void main()
{
    out_color = vec4(v_color, 1.0);
}
"""

if __name__ == "__main__":
    renderer = Renderer(800, 600, "OpenGL Window")

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


    shader = compileProgram(compileShader(VERTEX_SRC, GL_VERTEX_SHADER), compileShader(FRAGMENT_SRC, GL_FRAGMENT_SHADER))
    glUseProgram(shader)


    letter_l = Object(l_vertices, l_indices, pyrr.Matrix44.from_translation([0.5, -0.5, 0]), shader)
    letter_c = Object(c_vertices, c_indices, pyrr.Matrix44.from_translation([0.5, 0.5, 0]), shader)
    letter_a = Object(a_vertices, a_indices, pyrr.Matrix44.from_translation([-0.5, 0.5, 0]), shader)

    renderer.add_object(letter_l)
    renderer.add_object(letter_c)
    renderer.add_object(letter_a)
    
    renderer.object_rotation(5)
    renderer.object_rotation(10)
    renderer.object_rotation(20)

    renderer.run()
    renderer.cleanup()
