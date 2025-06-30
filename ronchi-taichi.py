#!/usr/bin/env python

import sys
import argparse

try:
    import taichi as ti
    import taichi.math as tm
except ImportError as e:
    print("You need taichi to run this version, try 'pip install taichi'")
    print(e)
    sys.exit(1)

ti.init(arch=ti.cpu)

# create an 800x800 image

RESOLUTION = 800
pixels = ti.field(dtype=float, shape=(RESOLUTION, RESOLUTION))

@ti.func
def reflect(I : ti.math.vec3, N : ti.math.vec3) -> ti.math.vec3:
    c = -2.0 * ti.math.dot(I, N) 
    R = c * N + I
    R = ti.math.normalize(R)
    return R

@ti.kernel
def paint(
    diameter: float,
    conic_constant: float,
    focal_length: float,
    grating_frequency: float,
    offset: float,
    invert: int
):

    O = ti.math.vec3(0., 0., 2.0 * focal_length + offset)

    for i, j in pixels:
        pixels[i, j] = i / RESOLUTION * j / RESOLUTION
        x = ((i + 0.5) / RESOLUTION - 0.5) * diameter
        y = ((j + 0.5) / RESOLUTION - 0.5) * diameter
        if x * x + y * y < diameter * diameter / 4:
            z = 0.
            if abs(conic_constant + 1.0) < 1e-5:
                z = (x * x + y * y) / (4.0 * focal_length)
            else:
                z = (2.0*focal_length - ti.math.sqrt((conic_constant + 1.0)*(-x*x - y*y) + 4.*focal_length*focal_length)) / (conic_constant + 1.)

            P = ti.math.vec3(x, y, z)
            I = P - O 
            I = ti.math.normalize(I)

            N = ti.math.vec3(-2.0 * x, -2.0 * y, 4.0 * focal_length - 2.0 * (conic_constant + 1.0) * z) 
            N = ti.math.normalize(N)

            R = reflect(I, N) ;

            t = (2.0 * focal_length + offset - z) / R[2] ;

            if t >= 0.:
                gx = (P[0] + t * R[0]) * 2.0 * grating_frequency - 0.5 ;
                if invert:
                    pixels[i, j] = 0. if int(ti.math.floor(gx)) & 1 else 1.
                else:
                    pixels[i, j] = 1. if int(ti.math.floor(gx)) & 1 else 0.
            else:
                pixels[i, j] = 0.5
        else:
            pixels[i, j] = 0.3


def main():
    invert = 0

    # setup some standard settings
    parser = argparse.ArgumentParser(description="Generate Ronchi test patterns.")
    parser.add_argument(
        "-d",
        "--diameter",
        type=float,
        help="Diameter (default: %(default)s)",
        default=12.5,
    )
    parser.add_argument(
        "-f",
        "--focal-length",
        type=float,
        help="Focal length (default: %(default)s)",
        default=62.5,
    )
    parser.add_argument(
        "-g",
        "--grating-frequency",
        type=float,
        help="Grating frequency (default: %(default)s)",
        default=85.0,
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=float,
        help="Offset (default: %(default)s)",
        default=-0.25,
    )
    parser.add_argument(
        "-k",
        "--conic-constant",
        type=float,
        help="Conic constant K (default: -1.0)",
        default=-1,
    )
    parser.add_argument(
        "--out",
        type=str,
        default="ronchi.jpg",
        help=f"Output filename (default: %(default)s). "
        "Pillow will infer format from extension (e.g., .pgm, .png).",
    )
    args = parser.parse_args()

    window = ti.ui.Window("Ronchi", res=(RESOLUTION, RESOLUTION))
    canvas = window.get_canvas()

    print(f"Diameter: {args.diameter:.2f}")
    print(f"Focal Length: {args.focal_length:.2f}")
    print(f"Conic Constant: {args.conic_constant:.1f}")
    print(f"Grating Frequency: {args.grating_frequency:.1f}")
    print(f"Grating Offset: {args.offset:.3f}")
    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.UP:
                args.offset += -0.01 
            elif e.key == ti.ui.DOWN:
                args.offset +=  0.01
            elif e.key == 'r':
                args.offset = -0.25
            elif e.key == 'i':
                invert = 1 - invert
        paint(
            args.diameter,
            args.conic_constant,
            args.focal_length,
            args.grating_frequency,
            args.offset,
            invert
        )
        canvas.set_image(pixels)

        label_str = f"{args.diameter:.1f} inch, {args.focal_length:.1f} FL, f/{args.focal_length/args.diameter:.2f}, {args.grating_frequency} lpi"

        gui = window.get_gui()
        with gui.sub_window("Parameters", x=0, y=0, width=0.35, height=0.08): # Create a sub-window for the text
            gui.text(label_str) # Display the label
            gui.text(f'Offset: {abs(args.offset):.3f} {"inside" if args.offset < 0 else "outside"} focus' ) # Display the offset
        window.show()


if __name__ == "__main__":
    main()
