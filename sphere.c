#include "GL/glew.h"
#include "GL/gl.h"
#include "GL/glut.h"
#include <cglm/cglm.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>  // gettimeofday
#include <math.h>
#include <assert.h>

#include <png.h>  // load .png texture image

#define MAX_INFO 512
// Sphere properties
#define N_STACKS 64
#define N_SECTORS 64
#define N_VERTICES ((N_STACKS + 1) * (N_SECTORS + 1))
#define N_INDICES (6 * N_SECTORS + 6 * (N_STACKS - 2) * N_SECTORS)

// Source of vertex shader in glsl
const GLchar *vertexShaderSource = R"glsl(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec3 aNormal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 normalMatrix;

out vec3 vertexColor;
out vec3 vertexNormal;
out vec3 fragPos;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    fragPos = vec3(model * vec4(aPos, 1.0));
    vertexColor = aColor;
    vertexNormal = vec3(normalMatrix * vec4(aNormal, 0.0));
}
)glsl";

// Source of fragment shader in glsl
const GLchar *fragmentShaderSource = R"glsl(
#version 460 core
out vec4 fragColor;

in vec3 vertexColor;
in vec3 vertexNormal;
in vec3 fragPos;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    
    vec3 lightDir = normalize(lightPos - fragPos);
    // ambient lighting
    float ambient = 0.1;
    vec3 ambientColor = ambient * lightColor;
    
    // diffuse lighting
    float diffuse = max(dot(lightDir, vertexNormal), 0.0);
    vec3 diffuseColor = diffuse * lightColor;
    
    // specular lighting
    float specularStrength = 0.8;
    
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, vertexNormal);
    float specular = pow(max(dot(viewDir, reflectDir), 0.0), 128);
    vec3 specularColor = specular * specularStrength * lightColor;
    
    // Final result
    vec3 result = (ambientColor + diffuseColor + specularColor) * vertexColor;
    fragColor = vec4(result, 1.0);
}
)glsl";

// Vertex attributes
typedef struct {
	vec3 pos;
	vec3 col;
	vec3 normal;
} Vertex;

// GLOBALS
GLuint shaderProgram;
GLuint vaoVert, vaoFloor;
GLsizei nFloorVertices = 0;
GLuint64 last;
vec3 sphereCenter = {0.0f, 0.0f, -1.5f};

GLfloat min(GLfloat a, GLfloat b) {
	return (a < b) ? a : b;
}

void pushModel(mat4 in, mat4 *out) {
	// Get uniform location of model matrix
	GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
	// Return previous model matrix
	if (out != NULL) {
		glGetUniformfv(shaderProgram, modelLoc, (GLfloat *)(*out));
	}
	// Update model matrix
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (GLfloat *)in);	
}

void draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    // Last argument specifies total number of vertices. Three consecutive
    // vertices are drawn as one triangle
    glBindVertexArray(vaoVert);
    glDrawElements(GL_TRIANGLES, N_INDICES, GL_UNSIGNED_INT, (void *)0);
    
    // Set model matrix back to identity
    //mat4 previous;
    //pushModel(GLM_MAT4_IDENTITY, &previous);
    
    glBindVertexArray(vaoFloor);
    glDrawArrays(GL_TRIANGLES, 0, nFloorVertices);
    // Reset model matrix to original value
    //pushModel(previous, NULL);
    
    // Swap buffers and show the buffer's content on the screen
    glutSwapBuffers();
}

void rotateModel(GLfloat angle) {
	// Get uniform location of model matrix
	GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
	// Get uniform location of normal matrix
	GLint normalLoc = glGetUniformLocation(shaderProgram, "normalMatrix");
	// Store current model matrix in mat4 variable
	mat4 model;
	glGetUniformfv(shaderProgram, modelLoc, (GLfloat *)model);
	// Store current normal matrix
	mat4 normalMatrix;
	glGetUniformfv(shaderProgram, normalLoc, (GLfloat *)normalMatrix);
	// Translate to origin
	glm_translate_z(model, -1.5f);
	// Rotation matrix
	mat4 rot = GLM_MAT4_IDENTITY_INIT;
	// Rotate current model matrix by 'angle' degrees
	glm_rotate_y(rot, glm_rad(angle), rot);
	glm_rotate_x(rot, glm_rad(angle), rot);
	// Multiply model matrix by rotation matrix
	glm_mat4_mul(model, rot, model);
	// Multiply normalMatrix by rotation matrix
	glm_mat4_mul(normalMatrix, rot, normalMatrix);
	// Translate back
	glm_translate_z(model, 1.5f);
	// Update model matrix
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (GLfloat *)model);
	// Update normal matrix to be the current rotation matrix
	glUniformMatrix4fv(normalLoc, 1, GL_FALSE, (GLfloat *)normalMatrix);
}

void rotateCamera(GLfloat angle) {
	// Fetch current camera position
	vec3 cameraPos;
	vec3 cameraUp = {0.0f, 1.0f, 0.0f};  // stays the same
	vec3 cameraDir;
	
	GLint cameraLoc = glGetUniformLocation(shaderProgram, "viewPos");
    glGetUniformfv(shaderProgram, cameraLoc, (GLfloat *)cameraPos);
    
    // Rotate camera position around y-axis (origin of sphere)
    glm_vec3_sub(cameraPos, sphereCenter, cameraPos);  // translate
    glm_vec3_rotate(cameraPos, angle, (vec3){0.0f, 1.0f, 0.0f});
    glm_vec3_add(cameraPos, sphereCenter, cameraPos);  // translate back
    // Update camera position
    glUniform3fv(cameraLoc, 1, (GLfloat *)cameraPos);
    
    // Camera direction always points to center of sphere
    glm_vec3_sub(sphereCenter, cameraPos, cameraDir);
    // Normalize direction
    glm_vec3_normalize(cameraDir);
    
    // Update view matrix
    mat4 view;
    GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
    glm_look(cameraPos, cameraDir, cameraUp, view);
    
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, (GLfloat *)view);
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
	const GLfloat speed = 0.0001f;
	// Rotation angle
	const GLfloat angle = speed * (GLfloat)deltaT;
	// Rotate scene
	//rotateModel(angle);
	// Rotate camera
	rotateCamera(angle);
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
    mat4 normalMatrix = GLM_MAT4_IDENTITY_INIT;
    
    // Initialize camera & light source position
    vec3 cameraPos = {0.0f, 0.0f, 2.0f};
    vec3 cameraDir = {0.0f, 0.0f, -1.0f};  // direction camera is looking in
    vec3 cameraUp  = {0.0f, 1.0f, 0.0f};
    
    vec3 lightPos = {0.0f, 10.0f, -1.5f};
    
    // Set camera & light position in shader
    GLint cameraLoc = glGetUniformLocation(shaderProgram, "viewPos");
    glUniform3fv(cameraLoc, 1, (GLfloat *)cameraPos);
    
    GLint lightLoc = glGetUniformLocation(shaderProgram, "lightPos");
    glUniform3fv(lightLoc, 1, (GLfloat *)lightPos);
    
    // Modify view matrix to fix camera
    glm_look(cameraPos, cameraDir, cameraUp, view);
    
	GLfloat width = glutGet(GLUT_WINDOW_WIDTH);
	GLfloat height = glutGet(GLUT_WINDOW_HEIGHT);
	mat4 proj;
    glm_perspective(glm_rad(60.0f), width / height, 0.5f, 200.0f, proj); 
	
    // Set model, view, projection and normal matrices
	GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
	glUniformMatrix4fv(modelLoc, 1, GL_FALSE, (GLfloat *)model);
	
	GLint viewLoc  = glGetUniformLocation(shaderProgram, "view");
	glUniformMatrix4fv(viewLoc, 1, GL_FALSE, (GLfloat *)view);
	
	GLint projLoc  = glGetUniformLocation(shaderProgram, "projection");
	glUniformMatrix4fv(projLoc, 1, GL_FALSE, (GLfloat *)proj);
	
	GLint normLoc = glGetUniformLocation(shaderProgram, "normalMatrix");
	glUniformMatrix4fv(normLoc, 1, GL_FALSE, (GLfloat *)normalMatrix);
}

// See http://www.songho.ca/opengl/gl_sphere.html for details
void initSphereProp(Vertex *vertices, GLuint *indices, 
	GLfloat radius, vec3 center) {
	
	GLfloat stackStep = GLM_PI / (GLfloat)N_STACKS;
	GLfloat sectorStep = 2.0f * GLM_PI / (GLfloat)N_SECTORS;
	GLfloat stackAngle, sectorAngle;
	GLfloat invLength = 1.0f / radius;
	
	GLfloat x, y, z, xy;
	GLfloat cx = center[0];
	GLfloat cy = center[1];
	GLfloat cz = center[2];
	
	// Initialize vertices
	GLuint i, j;
	for (i = 0; i <= N_STACKS; ++i) {
		
		stackAngle = GLM_PI / 2.0f - (GLfloat)i * stackStep;
		xy = radius * cosf(stackAngle);
		z  = radius * sinf(stackAngle);
		
		for (j = 0; j <= N_SECTORS; ++j) {
			
			sectorAngle = (GLfloat)j * sectorStep;
			
			x = xy * cosf(sectorAngle);
			y = xy * sinf(sectorAngle);
			
			// Add current vertex
			vertices[i * (N_SECTORS + 1) + j] = (Vertex) {
				.pos = {x + cx, y + cy, z + cz},
				.col = {1.0f, 165.0f/255.0f, 0.0f},  // orange
				.normal = {x * invLength, y * invLength, z * invLength}
			};
		}
	}
	
	// Initialize triangle indices
	GLuint index = 0;
	
	GLuint k1, k2;
	for (i = 0; i < N_STACKS; ++i) {
		
		k1 = i * (N_SECTORS + 1);  // beginning of current stack
		k2 = k1 + N_SECTORS  + 1;  // beginning of next stack
		
		for (j = 0; j < N_SECTORS; ++j, ++k1, ++k2) {
			
			// 2 triangles per sector (excluding first and last stack)
			// k1 => k2 => k1 + 1
			if (i != 0) {
				indices[index++] = k1;
				indices[index++] = k2;
				indices[index++] = k1 + 1;
			}
			
			// k1 + 1 => k2 => k2 + 1
			if (i != (N_STACKS - 1)) {
				indices[index++] = k1 + 1;
				indices[index++] = k2;
				indices[index++] = k2 + 1;
			}
		}
	}
	
	// DEBUG
	assert(index == N_INDICES);
}

void initFloorProp(Vertex **vertices, GLfloat spacing,
	vec2 start, vec2 end, GLfloat height, GLsizei *nVertices) {
	
	// Make sure nVertices and nIndices are not NULL
	assert(nVertices != NULL);
	// Make sure spacing is non-zero
	assert(spacing > 0.0f);
	// Make sure start & end correspond to bottom left and top right
	// corner respectively
	assert(start[0] < end[0] && start[1] < end[1]);
	
	// Length of floor in x/z direction
	const GLfloat lx = end[0] - start[0];
	const GLfloat lz = end[1] - start[1];
	// Number of vertices of floor in x/z direction
	const GLuint Nx = ceilf(lx / spacing) + 1;
	const GLuint Nz = ceilf(lz / spacing) + 1;
	const GLuint nTiles = (Nx - 1) * (Nz - 1);
	
	// Set number of vertices (2 triangles @ 6 vertices per tile)
	*nVertices = 6 * nTiles;
	
	GLuint ix, iz;
	
	// Allocate necessary memory
	*vertices = (Vertex *) malloc((*nVertices) * sizeof(Vertex));
	assert(*vertices != NULL);
	
	GLuint tileIndex = 0, vertexIndex = 0;
	GLfloat x, z, nextX, nextZ;
	// Initialize vertices
	for (iz = 0; iz < Nz - 1; ++iz) {
		
		z = min(iz * spacing + start[1], end[1]);
		nextZ = min(z + spacing, end[1]);
		
		for (ix = 0; ix < Nx - 1; ++ix) {
			
			x = min(ix * spacing + start[0], end[0]);
			nextX = min(x + spacing, end[0]);
			
			// Alternate color every other tile
			if (tileIndex % 2 == 0) {  // white color
				// bottom left corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {x, height, z},
					.col = {1.0f, 1.0f, 1.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// bottom right corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {nextX, height, z},
					.col = {1.0f, 1.0f, 1.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// top right corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {nextX, height, nextZ},
					.col = {1.0f, 1.0f, 1.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// bottom left corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {x, height, z},
					.col = {1.0f, 1.0f, 1.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// top right corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {nextX, height, nextZ},
					.col = {1.0f, 1.0f, 1.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// top left corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {x, height, nextZ},
					.col = {1.0f, 1.0f, 1.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
			} else {  // black color
				// bottom left corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {x, height, z},
					.col = {0.0f, 0.0f, 0.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// bottom right corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {nextX, height, z},
					.col = {0.0f, 0.0f, 0.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// top right corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {nextX, height, nextZ},
					.col = {0.0f, 0.0f, 0.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// bottom left corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {x, height, z},
					.col = {0.0f, 0.0f, 0.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// top right corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {nextX, height, nextZ},
					.col = {0.0f, 0.0f, 0.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
				// top left corner
				(*vertices)[vertexIndex++] = (Vertex) {
					.pos = {x, height, nextZ},
					.col = {0.0f, 0.0f, 0.0f},
					.normal = {0.0f, 1.0f, 0.0f}
				};
			}
			
			// Increment counter
			++tileIndex;
		}
		// Increment again to offset next row
		++tileIndex;
	}
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
	glClearColor(0.2, 0.2, 0.2, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_MULTISAMPLE);
	
	// Initialize vertex attributes and indices for sphere
	Vertex vertices[N_VERTICES];
	GLuint indices[N_INDICES];
	
	initSphereProp(vertices, indices, 1.0f, sphereCenter);
	
	// Initialize vertices of floor
	Vertex *floorVert = NULL;
	
	initFloorProp(&floorVert, 0.5f, (vec2){-6.0f, -7.5f},
		(vec2){6.0f, 4.5f}, -1.0f, &nFloorVertices);
	
	// Initialize vertex buffer object
	GLuint vboVert = initVBO((void *)vertices, GL_ARRAY_BUFFER, sizeof(vertices));
	GLuint vboFloor = initVBO((void *)floorVert, GL_ARRAY_BUFFER, nFloorVertices * sizeof(Vertex));
	
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
	glGenVertexArrays(1, &vaoVert);
	glGenVertexArrays(1, &vaoFloor);
	
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
	enableVertAttrib(vboVert, size, stride, 0, (void *)offsetof(Vertex, pos));
	// Enable vertex color attribute
	enableVertAttrib(vboVert, size, stride, 1, (void *)offsetof(Vertex, col));
	// Enable vertex normal attribute
	enableVertAttrib(vboVert, size, stride, 2, (void *)offsetof(Vertex, normal));
	
	glBindVertexArray(vaoFloor);
	
	// Enable vertex attributes of floor
	enableVertAttrib(vboFloor, size, stride, 0, (void *)offsetof(Vertex, pos));
	enableVertAttrib(vboFloor, size, stride, 1, (void *)offsetof(Vertex, col));
	enableVertAttrib(vboFloor, size, stride, 2, (void *)offsetof(Vertex, normal));
	
	// Cleanup data associated with floor
	free(floorVert);
	
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
