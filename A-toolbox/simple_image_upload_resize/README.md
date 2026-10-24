# Image Upload + Resize

Category `image`.

| Node | Does |
|---|---|
| **Simple Image Upload + Resize** | Upload one image, downscale to `max_megapixels` (sides divisible by 32; smaller images untouched). Outputs `image` and `mask`. |
| **Multi Image Upload + Resize (9)** | Drop up to nine images, resized together into one bundle. |
| **Image Set Splitter (9)** | Splits that bundle into `image_1`..`image_9`. |

## Mask output (Simple Image Upload + Resize)
`mask` follows Load Image: `1 - alpha`. Use a transparent PNG, or right-click the node, *Open in MaskEditor* and paint. An image without alpha gives an all-zero mask the size of the resized image.
