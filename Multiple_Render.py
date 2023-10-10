import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr


class Shapes:

    def __init__(self):
        self.cube = self.cube()
        self.pyramid = self.cube()

    def cube(self):
        cube_vertices = [
            -0.5, 0.5, -0.5, 1.0, 0.0, 0.0,  # 1
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

        return [cube_vertices, cube_indices]

    def pyramid(self):

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

        return [pyramid_vertices, pyramid_indices]


class Shaders:

    def __init__(self):
        self.vertex_src = """
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

        self.fragment_src = """
        # version 330

        in vec3 v_color;

        out vec4 out_color;

        void main()
        {
            out_color = vec4(v_color, 1.0);
        }
        """


class VAO:

    def __init__(self, shape_vertices, shape_indices):
        self.shape_vertices = shape_vertices
        self.shape_indices = shape_indices

    def make_vao(self):

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        # VBO
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, self.shape_vertices.nbytes,
                     self.shape_vertices, GL_STATIC_DRAW)

        # vertex
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        # color
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              24, ctypes.c_void_p(12))

        # indexing
        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.shape_indices.nbytes,
                     self.shape_indices, GL_STATIC_DRAW)
        glBindVertexArray(0)


class Transformation:

    def scale(self, x, y, z):
        scale = pyrr.Matrix44.from_scale(pyrr.Vector3([x, y, z]))

        return scale

    def translation(self, x, y, z):
        translation = pyrr.Matrix44.from_translation([x, y, z])

        return translation


class ShaderSpace:

    def __init__(self):
        self.shaders = Shaders()

    def shader_space(self):
        shader = compileProgram(compileShader(
            self.shaders.vertex_src, GL_VERTEX_SHADER), compileShader(self.shaders.fragment_src, GL_FRAGMENT_SHADER))
        glUseProgram(shader)
        return shader

    def model_loc(self):
        shader = self.shader_space()

        return glGetUniformLocation(shader, "model")


if __name__ == "__main__":
    pass
