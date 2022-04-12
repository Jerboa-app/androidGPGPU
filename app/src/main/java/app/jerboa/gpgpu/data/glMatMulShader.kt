package app.jerboa.gpgpu.data

/*
    Both shaders assume for and nxn block matrix a viewport of (0,0,n,n), with vertices:

        -1f,-1f,0f,
        1f,-1f,0f,
        1f,1f,0f,
        -1f,-1f,0f,
        -1f,1f,0f,
        1f,1f,0f

    and texture coordinates

    Vertex Shader:

        0f,0f,
        1f,0f,
        1f,1f,
        0f,0f,
        0f,1f,
        1f,1f

    For a 1-1 mapping between texture coordinates and the elements of the block matrix to multiply.

    Simply reads in the position and texture coordinate (location 0,1) and
    pass on the texture coordinate to the fragment shader.

    FragmentShader:

    The shader multiplies textures by their RGBA values as if they are 2x2 matrices. Each texture
    is one block in an nxn block matrix.

    The multiplication is achieved via the loop, see coordinates below:

        for (int k = 0; k < n; k++){\n"+
            "       float coordK = (float(k)+0.5)/fn;\n"+
            "       vec2 kj = vec2(o_texCoords.s,coordK);\n"+
            "       vec2 ik = vec2(coordK,o_texCoords.t);\n"+
            "       vec4 A = texture(textureX,ik.xy);\n"+
            "       vec4 B = texture(textureY,kj.xy);\n"+
            "       o_fragColor = o_fragColor + BlockMul(A,B);\n"+
            "   }\n"+

    and placed within the 0 attribute as output

    Recall openGL has textures indexed like (normed float coords):

    0,1 ---- 1,1
     |        |
     |        |
    0,0 ---- 1,0

    But a 2x2 matrix is typically indexed like (positive integers)

    _____
   |11|12|
   |21|22|
    -----

    To get to texture coords take indices starting at 0 instead of 1:

    _____
   |00|01|
   |10|11|
    -----

    Then see we must flip in y and permute the indices to make the code as
    simple as possible in the shader

 */


data class glMatMulShader(
    override val vertexShader: String =
        "#version 300 es\n"+
        "precision mediump float;\n"+
        "layout(location = 0) in vec3 position;\n"+
        "layout(location = 1) in vec2 i_texCoords;\n"+
        "out vec2 o_texCoords;\n"+
        "void main(void){\n" +
        "   gl_Position = vec4(position,1);\n"+
        "   o_texCoords = i_texCoords.st;\n"+
        "}\n",
    override val fragmentShader: String =
        "#version 300 es\n"+
        "precision mediump float;\n"+
        "in vec2 o_texCoords;\n"+
        "layout(location = 0) out vec4 o_fragColor;\n"+
        "uniform sampler2D textureX;\n"+
        "uniform sampler2D textureY;\n"+
        "uniform int n;\n"+
        "vec4 BlockMul(in vec4 A, in vec4 B){\n"+
        "   return vec4(A.r*B.r+A.g*B.b, A.r*B.g+A.g*B.a, A.b*B.r+A.a*B.b, A.b*B.g+A.a*B.a);\n"+
        "}\n"+
        "void main(void) {\n"+
        "   float fn = float(n);\n"+
        "   o_fragColor = vec4(0.0,0.0,0.0,0.0);\n"+
        "   for (int k = 0; k < n; k++){\n"+
        "       float coordK = (float(k)+0.5)/fn;\n"+
        "       vec2 kj = vec2(o_texCoords.s,coordK);\n"+
        "       vec2 ik = vec2(coordK,o_texCoords.t);\n"+
        "       vec4 A = texture(textureX,ik.xy);\n"+
        "       vec4 B = texture(textureY,kj.xy);\n"+
        "       o_fragColor = o_fragColor + BlockMul(A,B);\n"+
        "   }\n"+
        "}"
) : Shader(vertexShader, fragmentShader)