package app.jerboa.gpgpu.data

data class GPUData(
    val time: Long,
    val textureTime: Long,
    val drawTime: Long,
    val error: Long
)