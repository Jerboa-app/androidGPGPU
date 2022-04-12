package app.jerboa.gpgpu.gl

import app.jerboa.gpgpu.data.glMatMulShader
import app.jerboa.gpgpu.maths.matrixTo2x2BlockMatrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ceil
import kotlin.math.sqrt
import java.nio.FloatBuffer
import java.nio.IntBuffer
import kotlin.system.measureNanoTime
import android.opengl.GLES30 as gl3

fun matMul(x:Array<Float>, y:Array<Float>): Triple<FloatArray, Long, Long> {
    var timeMem: Long = 0
    var timeDraw: Long = 0
    var clockMemStart = System.currentTimeMillis()

    val A = matrixTo2x2BlockMatrix(x)
    val B = matrixTo2x2BlockMatrix(y)
    var n = ceil(sqrt(B.size.toDouble() / 4f)).toInt()
    // x
    val xTexBuffer = IntBuffer.allocate(1)
    gl3.glGenTextures(1, xTexBuffer)
    xTexBuffer.flip()
    xTexBuffer.limit(1)
    val xTex = xTexBuffer[0]
    // y
    val yTexBuffer = IntBuffer.allocate(2)
    yTexBuffer.flip()
    yTexBuffer.limit(2)
    gl3.glGenTextures(2, yTexBuffer)
    val yWrite = yTexBuffer[0]
    val yRead = yTexBuffer[1]
    glError()

    // get texture data
    var dataBufferX = FloatBuffer.allocate(B.size)
    dataBufferX.put(B.toFloatArray())

    var dataBufferY = FloatBuffer.allocate(B.size)
    dataBufferY.put(B.toFloatArray())

    val returnBuffer = FloatBuffer.allocate(B.size)
    returnBuffer.put(FloatArray(B.size) { -1f })

    // init textures and transfer data
    initTexture2DRGBA32F(xTex, n)
    transferToTexture2DRGBA32F(xTex, dataBufferX, n)
    initTexture2DRGBA32F(yWrite, n)
    transferToTexture2DRGBA32F(yWrite, dataBufferY, n)
    initTexture2DRGBA32F(yRead, n)
    transferToTexture2DRGBA32F(yRead, returnBuffer, n)

    timeMem = System.currentTimeMillis() - clockMemStart

    // compile the shaders
    val glslProg: Int = gl3.glCreateProgram()
    compileGLSLProgram(glslProg, glMatMulShader().vertexShader, glMatMulShader().fragmentShader)

    // get the uniforms
    val xParam = gl3.glGetUniformLocation(glslProg, "textureX")
    val yParam = gl3.glGetUniformLocation(glslProg, "textureY")
    val nParam = gl3.glGetUniformLocation(glslProg, "n")
    glError()

    clockMemStart = System.currentTimeMillis()
    // create and bind a frame buffer
    val fboBuffer = IntBuffer.allocate(1)
    gl3.glGenFramebuffers(1, fboBuffer)
    fboBuffer.flip()
    fboBuffer.limit(1)
    val fbo = fboBuffer[0]
    gl3.glBindFramebuffer(gl3.GL_FRAMEBUFFER, fbo)
    glError()
    glBufferStatus()

    gl3.glFramebufferTexture2D(
        gl3.GL_FRAMEBUFFER,
        gl3.GL_COLOR_ATTACHMENT0,
        gl3.GL_TEXTURE_2D,
        yWrite,
        0
    )

    glError()
    glBufferStatus()

    gl3.glFramebufferTexture2D(
        gl3.GL_FRAMEBUFFER,
        gl3.GL_COLOR_ATTACHMENT1,
        gl3.GL_TEXTURE_2D,
        yRead,
        0
    )

    glError()
    glBufferStatus()

    // use program
    gl3.glUseProgram(glslProg)

    // set xTex
    gl3.glActiveTexture(gl3.GL_TEXTURE1)
    gl3.glBindTexture(gl3.GL_TEXTURE_2D, xTex)
    gl3.glUniform1i(xParam, 1)
    gl3.glUniform1i(nParam, n)

    glError()

    // set the draw buffer
    val drawBuffers = IntBuffer.allocate(1)
    drawBuffers.put(gl3.GL_COLOR_ATTACHMENT0)
    drawBuffers.flip()
    drawBuffers.limit(1)
    gl3.glDrawBuffers(1, drawBuffers)
    glError()
    glBufferStatus()

    // set yTex
    gl3.glActiveTexture(gl3.GL_TEXTURE0)
    gl3.glBindTexture(gl3.GL_TEXTURE_2D, yRead)
    gl3.glUniform1i(yParam, 1)

    glError()

    val verts: FloatBuffer = ByteBuffer.allocateDirect(6*3 * Float.SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        //FloatBuffer.allocate(6 * 3)
    verts.put(
        listOf<Float>(
            -1f, -1f, 0f,
            1f, -1f, 0f,
            1f, 1f, 0f,
            -1f, -1f, 0f,
            -1f, 1f, 0f,
            1f, 1f, 0f
        ).toFloatArray()
    )
    verts.flip()
    verts.limit(6 * 3)

    val texCoords: FloatBuffer = ByteBuffer.allocateDirect(6*2 * Float.SIZE_BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
    texCoords.put(
        listOf<Float>(
            0f, 0f,
            1f, 0f,
            1f, 1f,
            0f, 0f,
            0f, 1f,
            1f, 1f
        ).toFloatArray()
    )
    texCoords.flip()
    texCoords.limit(6 * 2)

    gl3.glViewport(0, 0, n, n)

    gl3.glEnableVertexAttribArray(0)
    gl3.glVertexAttribPointer(0, 3, gl3.GL_FLOAT, false, 0, verts)
    glError()

    gl3.glEnableVertexAttribArray(1)
    gl3.glVertexAttribPointer(1, 2, gl3.GL_FLOAT, false, 0, texCoords)
    glError()

    timeMem += System.currentTimeMillis()-clockMemStart

    timeDraw = measureNanoTime {
        gl3.glDrawArrays(gl3.GL_TRIANGLES, 0, 6)
    }
    gl3.glFinish()
    glError()


    glBufferStatus()

    clockMemStart = System.currentTimeMillis()

    gl3.glReadBuffer(gl3.GL_COLOR_ATTACHMENT0)
    returnBuffer.flip()
    returnBuffer.limit(B.size)

    glBufferStatus()
    glError()
    gl3.glReadPixels(
        0,
        0,
        n,
        n,
        gl3.GL_RGBA,
        gl3.GL_FLOAT,
        returnBuffer
    )
    glError()
    glBufferStatus()

    returnBuffer.flip()
    returnBuffer.limit(B.size)

    gl3.glDeleteTextures(2, yTexBuffer)
    gl3.glDeleteTextures(1, xTexBuffer)
    gl3.glDeleteFramebuffers(1, fboBuffer)
    gl3.glDeleteProgram(glslProg)

    xTexBuffer.clear()
    yTexBuffer.clear()
    fboBuffer.clear()

    val ret: FloatArray = FloatArray(x.size) { 0f }
    for (i in x.indices){
        ret[i] = returnBuffer[i]
    }
    returnBuffer.clear()

    timeMem += System.currentTimeMillis()-clockMemStart

    return Triple<FloatArray,Long,Long>(ret,timeMem,timeDraw)
}