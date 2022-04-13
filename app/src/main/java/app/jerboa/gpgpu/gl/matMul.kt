package app.jerboa.gpgpu.gl

import android.graphics.SurfaceTexture
import android.opengl.EGL14.EGL_CONTEXT_CLIENT_VERSION
import android.opengl.EGL14.EGL_OPENGL_ES2_BIT
import android.opengl.GLUtils
import app.jerboa.gpgpu.data.glMatMulShader
import app.jerboa.gpgpu.maths.BlockMatrix2x2ToMatrix
import app.jerboa.gpgpu.maths.matrixTo2x2BlockMatrix
import app.jerboa.gpgpu.maths.printMatrix
import app.jerboa.gpgpu.maths.trimMatrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGL10
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.egl.EGLContext
import kotlin.math.ceil
import kotlin.math.sqrt
import kotlin.system.measureNanoTime
import android.opengl.GLES30 as gl3

/*
    gl routines to multiply matrices x and y

    Handles all gl memory management etc.

    Output matrices need to be conver
 */
fun matMul(x:Array<Float>, y:Array<Float>): Triple<FloatArray, Long, Long> {
    // with thanks https://stackoverflow.com/questions/18529021/android-initialise-opengl2-0-context-with-egl/18537383#18537383
    // this sets up a context to render gl stuff off screen
    val mEgl = EGLContext.getEGL() as EGL10

    val mEglDisplay = mEgl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY)

    if (mEglDisplay === EGL10.EGL_NO_DISPLAY) throw RuntimeException(
        "Error: eglGetDisplay() Failed " + GLUtils.getEGLErrorString(
            mEgl.eglGetError()
        )
    )

    val version = IntArray(2)

    if (!mEgl.eglInitialize(
            mEglDisplay,
            version
        )
    ) throw RuntimeException("Error: eglInitialize() Failed " + GLUtils.getEGLErrorString(mEgl.eglGetError()))

    val maEGLconfigs = arrayOfNulls<EGLConfig>(1)

    val configsCount = IntArray(1)
    val configSpec = intArrayOf(
        EGL10.EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL10.EGL_RED_SIZE, 8,
        EGL10.EGL_GREEN_SIZE, 8,
        EGL10.EGL_BLUE_SIZE, 8,
        EGL10.EGL_ALPHA_SIZE, 8,
        EGL10.EGL_DEPTH_SIZE, 0,
        EGL10.EGL_STENCIL_SIZE, 0,
        EGL10.EGL_NONE
    )
    require(
        !(!mEgl.eglChooseConfig(
            mEglDisplay,
            configSpec,
            maEGLconfigs,
            1,
            configsCount
        ) || configsCount[0] == 0)
    ) { "Error: eglChooseConfig() Failed " + GLUtils.getEGLErrorString(mEgl.eglGetError()) }

    if (maEGLconfigs.get(0) == null) throw RuntimeException("Error: eglConfig() not Initialized")

    val attrib_list = intArrayOf(EGL_CONTEXT_CLIENT_VERSION, 2, EGL10.EGL_NONE)

    val mEglContext =
        mEgl.eglCreateContext(mEglDisplay, maEGLconfigs.get(0), EGL10.EGL_NO_CONTEXT, attrib_list)

    val surfaceTexture = SurfaceTexture(0)
    surfaceTexture.setDefaultBufferSize(1,1);

    val mEglSurface = mEgl.eglCreateWindowSurface(mEglDisplay, maEGLconfigs[0], surfaceTexture, null);

    if (mEglSurface == null || mEglSurface == EGL10.EGL_NO_SURFACE)
    {
        val error = mEgl.eglGetError();

        if (error == EGL10.EGL_BAD_NATIVE_WINDOW)
        {
            println("Error: createWindowSurface() Returned EGL_BAD_NATIVE_WINDOW.")
        }
    }
    if (!mEgl.eglMakeCurrent(mEglDisplay, mEglSurface, mEglSurface, mEglContext))
        println("Make current failed")
    // end of context setup

    var timeMem: Long = 0
    var timeDraw: Long = 0
    var clockMemStart = System.currentTimeMillis()

    // convert to block form
    val A = matrixTo2x2BlockMatrix(x)
    val B = matrixTo2x2BlockMatrix(y)
    var n = ceil(sqrt(B.size.toDouble() / 4f)).toInt()
    println("A size gl = "+A.size.toString()+" n gl = $n")
    // x
    // careful of the byte order here, this form is necessary
    val xTexBuffer = ByteBuffer.allocateDirect(1 * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
    // generate a new texture handle
    gl3.glGenTextures(1, xTexBuffer)
    xTexBuffer.flip()
    xTexBuffer.limit(1)
    // grab the actual texture handle
    val xTex = xTexBuffer[0]
    // y, same again but with 2 textures
    val yTexBuffer = ByteBuffer.allocateDirect(2 * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
    yTexBuffer.flip()
    yTexBuffer.limit(2)
    gl3.glGenTextures(2, yTexBuffer)
    val yWrite = yTexBuffer[0]
    val yRead = yTexBuffer[1]
    glError()

    // get texture data into buffers
    var dataBufferX = ByteBuffer.allocateDirect(B.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    dataBufferX.put(B.toFloatArray())

    var dataBufferY = ByteBuffer.allocateDirect(B.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    dataBufferY.put(B.toFloatArray())

    val returnBuffer = ByteBuffer.allocateDirect(B.size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    returnBuffer.put(FloatArray(B.size) { -1f })

    // init textures and transfer data, one RGBAF32 === one block in the matrix!
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

    // grab the uniforms handles
    val xParam = gl3.glGetUniformLocation(glslProg, "textureX")
    val yParam = gl3.glGetUniformLocation(glslProg, "textureY")
    val nParam = gl3.glGetUniformLocation(glslProg, "n")
    glError()

    clockMemStart = System.currentTimeMillis()
    // create and bind a frame buffer
    val fboBuffer = ByteBuffer.allocateDirect(1 * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
    gl3.glGenFramebuffers(1, fboBuffer)
    fboBuffer.flip()
    fboBuffer.limit(1)
    val fbo = fboBuffer[0]
    gl3.glBindFramebuffer(gl3.GL_FRAMEBUFFER, fbo)
    glError()
    glBufferStatus()

    // create frame buffer textures in 0
    gl3.glFramebufferTexture2D(
        gl3.GL_FRAMEBUFFER,
        gl3.GL_COLOR_ATTACHMENT0,
        gl3.GL_TEXTURE_2D,
        yWrite,
        0
    )

    glError()
    glBufferStatus()
    // and in 1
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

    // set xTex bound to sampler for 1
    gl3.glActiveTexture(gl3.GL_TEXTURE1)
    gl3.glBindTexture(gl3.GL_TEXTURE_2D, xTex)
    gl3.glUniform1i(xParam, 1)
    gl3.glUniform1i(nParam, n)

    glError()

    // set the draw buffer
    val drawBuffers = ByteBuffer.allocateDirect(1 * 4).order(ByteOrder.nativeOrder()).asIntBuffer()
    // we are drawing into 0
    drawBuffers.put(gl3.GL_COLOR_ATTACHMENT0)
    drawBuffers.flip()
    drawBuffers.limit(1)
    gl3.glDrawBuffers(1, drawBuffers)
    glError()
    glBufferStatus()

    // set yTex bound to sampler for 0
    gl3.glActiveTexture(gl3.GL_TEXTURE0)
    gl3.glBindTexture(gl3.GL_TEXTURE_2D, yRead)
    gl3.glUniform1i(yParam, 1)

    glError()

    // a single quad (square)
    val verts: FloatBuffer = ByteBuffer.allocateDirect(6*3 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
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

    // map each corner in the square (verts) to a corner in the acutal texture (want a 1-1 mapping)
    val texCoords: FloatBuffer = ByteBuffer.allocateDirect(6*2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
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

    // view all n by n "pixels"
    gl3.glViewport(0, 0, n, n)

    // upload the vertices into attribute 0 (see shaders)
    gl3.glEnableVertexAttribArray(0)
    gl3.glVertexAttribPointer(0, 3, gl3.GL_FLOAT, false, 0, verts)
    glError()

    // textures go ini attribute 1
    gl3.glEnableVertexAttribArray(1)
    gl3.glVertexAttribPointer(1, 2, gl3.GL_FLOAT, false, 0, texCoords)
    glError()

    timeMem += System.currentTimeMillis()-clockMemStart
    // Drawing === Computing
    timeDraw = measureNanoTime {
        gl3.glDrawArrays(gl3.GL_TRIANGLES, 0, 6)
    }
    glError()


    glBufferStatus()

    clockMemStart = System.currentTimeMillis()

    // read from the drawn to attachment
    gl3.glReadBuffer(gl3.GL_COLOR_ATTACHMENT0)
    returnBuffer.flip()
    returnBuffer.limit(B.size)

    glBufferStatus()
    glError()
    // get all pixels in the return buffer
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

    // clean up gl stuff
    gl3.glDeleteTextures(2, yTexBuffer)
    gl3.glDeleteTextures(1, xTexBuffer)
    gl3.glDeleteFramebuffers(1, fboBuffer)
    gl3.glDeleteProgram(glslProg)
    // cleanup buffers
    xTexBuffer.clear()
    yTexBuffer.clear()
    fboBuffer.clear()

    // get result as float array
    var ret: FloatArray = FloatArray(B.size) { 0f }
    for (i in B.indices){
        ret[i] = returnBuffer[i]
    }

    returnBuffer.clear()

    // convert back from block, then trim the zeros, if there are any
    ret = trimMatrix(BlockMatrix2x2ToMatrix(ret.toTypedArray()),ceil(sqrt(x.size.toDouble())).toInt()).toFloatArray()

    timeMem += System.currentTimeMillis()-clockMemStart
    // done!
    return Triple<FloatArray,Long,Long>(ret,timeMem,timeDraw)
}