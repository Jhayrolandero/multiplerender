import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import numpy as np

# Define vertices and indices
vertices = np.array([
    [-1, -1, 0],  # Vertex 0
    [1, -1, 0],   # Vertex 1
    [0, 1, 0],    # Vertex 2
], dtype=np.float32)

indices = np.array([
    0, 1, 2,  # Triangle
], dtype=np.uint32)

def draw_triangle():
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)
    glDisableClientState(GL_VERTEX_ARRAY)

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)  # Rotate the triangle
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_triangle()
        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__":
    # Create a VBO for vertices
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Create a VBO for indices
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    main()
