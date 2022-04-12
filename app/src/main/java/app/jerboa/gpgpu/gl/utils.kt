package app.jerboa.gpgpu.gl

import java.nio.FloatBuffer
import android.opengl.GLES30 as gl3

fun glBufferStatus(): Int {
    val e = gl3.glCheckFramebufferStatus(gl3.GL_FRAMEBUFFER)
    when(e){
        gl3.GL_FRAMEBUFFER_UNDEFINED -> {println("GL_FRAMEBUFFER_UNDEFINED")}
        gl3.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT -> {println("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT")}
        gl3.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT -> {println("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT")}
        gl3.GL_FRAMEBUFFER_UNSUPPORTED -> {println("GL_FRAMEBUFFER_UNSUPPORTED")}
        gl3.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE -> {println("GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE")}
    }
    return e
}

fun glError(): Int {
    val e = gl3.glGetError()
    when (e){
        gl3.GL_NO_ERROR -> {}
        gl3.GL_INVALID_ENUM -> {println("GL_INVALID_ENUM")}
        gl3.GL_INVALID_VALUE -> {println("GL_INVALID_VALUE")}
        gl3.GL_INVALID_OPERATION -> {println("GL_INVALID_OPERATION")}
        gl3.GL_OUT_OF_MEMORY -> {println("GL_OUT_OF_MEMORY")}
        gl3.GL_INVALID_FRAMEBUFFER_OPERATION -> {println("INVALID_FRAMEBUFFER_OPERATION")}
    }
    return e
}

fun compileGLSLProgram(id: Int, vert: String, frag: String){

    val vertexShader = gl3.glCreateShader(gl3.GL_VERTEX_SHADER)
    gl3.glShaderSource(vertexShader,vert)
    gl3.glCompileShader(vertexShader)
    print("Vertex shader compiled ")
    println(gl3.glGetShaderInfoLog(vertexShader))
    gl3.glAttachShader(id,vertexShader)
    gl3.glLinkProgram(id)

    val fragmentShader = gl3.glCreateShader(gl3.GL_FRAGMENT_SHADER)
    gl3.glShaderSource(fragmentShader,frag)
    gl3.glCompileShader(fragmentShader)
    print("Fragment shader compiled ")
    println(gl3.glGetShaderInfoLog(fragmentShader))
    gl3.glAttachShader(id,fragmentShader)
    gl3.glLinkProgram(id)
    print("GLSL program linked ")
    println(gl3.glGetProgramInfoLog(id))
}

fun initTexture2DRGBA32F(id: Int, n: Int): Int {
    gl3.glBindTexture(gl3.GL_TEXTURE_2D,id)
    gl3.glTexParameteri(
        gl3.GL_TEXTURE_2D,
        gl3.GL_TEXTURE_MIN_FILTER,
        gl3.GL_NEAREST
    )
    gl3.glTexParameteri(
        gl3.GL_TEXTURE_2D,
        gl3.GL_TEXTURE_MAG_FILTER,
        gl3.GL_NEAREST
    )
    gl3.glTexParameteri(
        gl3.GL_TEXTURE_2D,
        gl3.GL_TEXTURE_WRAP_S,
        gl3.GL_CLAMP_TO_EDGE
    )
    gl3.glTexParameteri(
        gl3.GL_TEXTURE_2D,
        gl3.GL_TEXTURE_WRAP_T,
        gl3.GL_CLAMP_TO_EDGE
    )
    gl3.glTexImage2D(
        gl3.GL_TEXTURE_2D,
        0,
        gl3.GL_RGBA32F,
        n,
        n,
        0,
        gl3.GL_RGBA,
        gl3.GL_FLOAT,
        null
    )
    return glError()
}

fun transferToTexture2DRGBA32F(id: Int, data: FloatBuffer, n: Int): Int{
    data.flip()
    data.limit(n*n)
    gl3.glBindTexture(gl3.GL_TEXTURE_2D,id)

    gl3.glTexImage2D(
        gl3.GL_TEXTURE_2D,
        0,
        gl3.GL_RGBA32F,
        n,
        n,
        0,
        gl3.GL_RGBA,
        gl3.GL_FLOAT,
        data
    )
    gl3.glTexImage2D(
        gl3.GL_TEXTURE_2D,
        0,
        gl3.GL_LUMINANCE,
        n,
        n,
        0,
        gl3.GL_LUMINANCE,
        gl3.GL_FLOAT,
        data
    )
    return glError()
}

