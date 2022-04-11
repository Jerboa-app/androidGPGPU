package app.jerboa.gpgpu.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.Shapes
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Outline
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.Density
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.dp

/*

    Creates a triangular shape to clip images too or
    anything else compose does with shape!

    E.g:

    Image(
        painter = painterResource( id = img.id),
        contentDescription = "Image",
        modifier = Modifier
            .rotate(180f)
            .size(40.dp)
            .clip(
                DTriangle(with(LocalDensity.current) { 32.dp.toPx() })
        )
    )

 */
class DTriangle(private val s: Float): Shape {
    private val a: Float = 0.5f * s
    private val b: Float = 0.8660254037844386f*s
    override fun createOutline(
        size: Size,
        layoutDirection: LayoutDirection,
        density: Density,
    ) = Outline.Generic(Path().apply {
        moveTo(size.width/2f-a,size.height/2f-b/2f);
        lineTo(size.width/2f+a,size.height/2f-b/2f);
        lineTo(size.width/2f,size.height/2f+b/2f);
        lineTo(size.width/2f-a,size.height/2f-b/2f);
        close()
        // build your path here depending on params and size
    })
}

val Shapes = Shapes(
    small = RoundedCornerShape(4.dp),
    medium = RoundedCornerShape(4.dp),
    large = RoundedCornerShape(0.dp)
)