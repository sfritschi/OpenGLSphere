#include "GL/glew.h"
#include "GL/gl.h"
#include "GL/glut.h"
#include <cglm/cglm.h>

#include <stdio.h>
#include <sys/time.h>  // gettimeofday
#include <math.h>

#define MAX_INFO 512
#define N_VERTICES 8

// Source of vertex shader in glsl
const GLchar *vertexShaderSource = R"glsl(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 vertexColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    vertexColor = aColor;
}
)glsl";

// Source of fragment shader in glsl
const GLchar *fragmentShaderSource = R"glsl(
#version 460 core
out vec4 fragColor;

in vec3 vertexColor;

void main() {
    fragColor = vec4(vertexColor, 1.0);
}
)glsl";

// Vertex attributes
typedef struct {
	vec3 pos;
	vec3 col;
} Vertex;

// GLOBALS
GLuint shaderProgram;
GLsizei nIndices = 0;
GLuint64 last;

void draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Last argument specifies total number of vertices. Three consecutive
    // vertices are drawn as one triangle
    glDrawElements(GL_TRIANGLES, nIndices, GL_UNSIGNED_SHORT, (void *)0);
        
    // Swap buffers and show the buffer's content on the screen
    glutSwapBuffers();
}

void rotateModel(GLfloat angle) {
	// Get uniform location of model matrix
	GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
	// Store current model matrix in mat4 variable
	mat4 model;
	glGetUniformfv(shaderProgram, modelLoc, (GLfloat *)model);
	// Translate to origin
	glm_translate_z(model, -1.5f);
	// Rotate current model matrix by 'angle' degrees
	glm_rotate_y(model, glm_rad(angle), model);
	glm_rotate_x(model, glm_rad(angle), model);
	// Translate back
	glm_translate_z(model, 1.5f);
	// Update model matrix
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (GLfloat *)model);
}

GLuint64 getCurrentTime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	
	return (GLuint64)(tv.tv_sec * 1000) + (GLuint64)(tv.tv_usec / 1000);
}

void idleRotate() {
	const GLuint64 now = getCurrentTime();
	// Elapsed time since last call
	const GLuint64 deltaT = now - last;
	// Update last time
	last = now;
	// Rotation speed (45 deg / s = 0.045 deg / ms)
	const GLfloat speed = 0.0225f;
	// Rotation angle
	const GLfloat angle = speed * (GLfloat)deltaT;
	// Rotate scene
	rotateModel(angle);
	// Redisplay scene
	glutPostRedisplay(); 
}

void keyPressed(unsigned char key, int x, int y) {
    // Unused parameters; don't warn
    (void)x;
    (void)y;
    
    if (key == 'q') {
        exit(EXIT_SUCCESS);
    }
}

GLuint initVBO(const void *data, GLenum target, const GLsizeiptr size) {
    GLuint vbo;
    // Generate vbo
    glGenBuffers(1, &vbo);
    // Bind vbo as current buffer
    glBindBuffer(target, vbo);
    // Set buffer data for vbo
    glBufferData(target, size, data, GL_STATIC_DRAW);
    
    return vbo;
}

GLuint createShader(GLenum shaderType, const char *shaderSource) {
    // Create shader
    GLuint shader = glCreateShader(shaderType);
    // Specify source for shader
    glShaderSource(shader, 1, &shaderSource, NULL);
    // Compile
    glCompileShader(shader);
    // Check for successful compilation
    GLint success;
	GLchar infoLog[MAX_INFO];
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	
	if (!success) {
		glGetShaderInfoLog(shader, MAX_INFO, NULL, infoLog);
		fprintf(stderr, "SHADER COMPILATION FAILED:\n%s\n", infoLog);
        // Cleanup
		glDeleteShader(shader);
		exit(EXIT_FAILURE);
	}
    return shader;    
}

void enableVertAttrib(GLuint vbo, GLint size, GLsizeiptr stride, GLuint index, const void *offset) {
    // Bind vbo as current buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Initialize & enable vertex attribute
    glVertexAttribPointer(index, size, GL_FLOAT, GL_FALSE, stride, offset);
    // Enable vertex attribute array at index location
    glEnableVertexAttribArray(index);
}

void initScene() {
    // Manage scene
	mat4 model = GLM_MAT4_IDENTITY_INIT;  // identity
	mat4 view  = GLM_MAT4_IDENTITY_INIT;
    
	GLfloat width = glutGet(GLUT_WINDOW_WIDTH);
	GLfloat height = glutGet(GLUT_WINDOW_HEIGHT);
	mat4 proj;
    glm_perspective(glm_rad(90.0f), width / height, 0.5f, 200.0f, proj); 
	
    // Set model, view and projection matrices
	GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (GLfloat *)model);
	GLint viewLoc  = glGetUniformLocation(shaderProgram, "view");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, (GLfloat *)view);
	GLint projLoc  = glGetUniformLocation(shaderProgram, "projection");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, (GLfloat *)proj);
}

int main(int argc, char *argv[]) {
	
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(1000, 1000);
    glutInitWindowPosition(460, 0);
    glutCreateWindow("Sphere");
    
	GLenum err = glewInit();
	
	if (err != GLEW_OK) {
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(EXIT_FAILURE);	
	}
	glClearColor(0.8, 0.8, 0.8, 1.0);
	glEnable(GL_DEPTH_TEST);
	
	// Initialize data
	Vertex vertices[N_VERTICES] = {
		// Front face (green)
		(Vertex){.pos = {-0.5f, -0.5f, -1.0f}, .col = {0.1f, 0.8f, 0.6f}},
		(Vertex){.pos = {0.5f, -0.5f, -1.0f}, .col = {0.1f, 0.8f, 0.6f}},
		(Vertex){.pos = {0.5f, 0.5f, -1.0f}, .col = {0.1f, 0.8f, 0.6f}},
		(Vertex){.pos = {-0.5f, 0.5f, -1.0f}, .col = {0.1f, 0.8f, 0.6f}},
		// Back face (blue)
		(Vertex){.pos = {-0.5f, -0.5f, -2.0f}, .col = {0.1f, 0.8f, 0.6f}},
		(Vertex){.pos = {0.5f, -0.5f, -2.0f}, .col = {0.1f, 0.8f, 0.6f}},
		(Vertex){.pos = {0.5f, 0.5f, -2.0f}, .col = {0.1f, 0.8f, 0.6f}},
		(Vertex){.pos = {-0.5f, 0.5f, -2.0f}, .col = {0.1f, 0.8f, 0.6f}}
	};
	
	// Triangle indices
	GLushort indices[] = {
		// Front face
		0, 1, 2,
		0, 2, 3,
		// Back face
		4, 5, 6,
		4, 6, 7,
		// Bottom face
		0, 1, 5,
		0, 5, 4,
		// Top face
		3, 2, 6,
		3, 6, 7,
		// Left face
		4, 0, 3,
		4, 3, 7,
		// Right face
		1, 5, 6,
		1, 6, 2
	};
	// Set total number of indices
	nIndices = sizeof(indices) / sizeof(GLushort);
	
	// Initialize vertex buffer object
	GLuint vboVert = initVBO((void *)vertices, GL_ARRAY_BUFFER, sizeof(vertices));
	
	// Create shader program
	GLuint vertShader = createShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragShader = createShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
	
	shaderProgram = glCreateProgram();
	// Attach shaders to program & link
	glAttachShader(shaderProgram, vertShader);
	glAttachShader(shaderProgram, fragShader);
	
	glLinkProgram(shaderProgram);
	
	// Check for success
	GLint success;
    GLchar infoLog[MAX_INFO];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	
	if (!success) {
		glGetProgramInfoLog(shaderProgram, MAX_INFO, NULL, infoLog);
		fprintf(stderr, "SHADER PROGRAM LINKING FAILED:\n%s\n", infoLog);
        // Cleanup
		glDetachShader(shaderProgram, vertShader);
		glDetachShader(shaderProgram, fragShader);
		glDeleteShader(vertShader);
		glDeleteShader(fragShader);
		glDeleteProgram(shaderProgram);
		
		exit(EXIT_FAILURE);
	}
	
	// Initialize vertex array object
	GLuint vaoVert;
	glGenVertexArrays(1, &vaoVert);
	
	// Initialize index buffer object (element array)
	GLuint vboInd;
	glGenBuffers(1, &vboInd);
		
	glBindVertexArray(vaoVert);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboInd);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), 
		(void *)indices, GL_STATIC_DRAW);
	
	GLsizeiptr size = 3;  // size of individual attribute (3D)
	GLsizeiptr stride = sizeof(Vertex);
	// Enable vertex position attribute
	enableVertAttrib(vboVert, size, stride, 0, (void *)0);
	// Enable vertex color attribute
	enableVertAttrib(vboVert, size, stride, 1, (void *)offsetof(Vertex, col));
	
	// Start using shader program
	glUseProgram(shaderProgram);
	// Detach & delete shaders (not needed anymore)
	glDetachShader(shaderProgram, vertShader);
	glDetachShader(shaderProgram, fragShader);
	glDeleteShader(vertShader);
	glDeleteShader(fragShader);
	
	// Initialize scene
	initScene();
	
	// Initialize starting time
	last = getCurrentTime();
	
	// Specify render function
	glutDisplayFunc(draw);
	// Close window when key 'q' is pressed
	glutKeyboardFunc(keyPressed);
	// Animation (rotation) function
	glutIdleFunc(idleRotate);
	
	// Start program loop
	glutMainLoop();
	
	return EXIT_SUCCESS;
}
