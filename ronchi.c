#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>

/*
 *                       _     _ 
 *  _ __ ___  _ __   ___| |__ (_)
 * | '__/ _ \| '_ \ / __| '_ \| |
 * | | | (_) | | | | (__| | | | |
 * |_|  \___/|_| |_|\___|_| |_|_|
 *                               
 * A program for generating Ronchi test
 * patterns using elementary geometric 
 * optics.
 * 
 * Written by Mark VandeWettering, based
 * on code first written in 2001, but updated in 2015.
 */

typedef double Vec[3] ;

#define RESOLUTION	(2048)

#define VecDot(a,b) ((a)[0]*(b)[0]+(a)[1]*(b)[1]+(a)[2]*(b)[2])
#define VecLen(a)   sqrtf(VecDot(a,a))

double 	diameter = 6.0 ;
double 	flen = 24.0 ;
double	offset = -0.25 ;
double	grating = 100.0 ;
double 	K = -1.0 ;

/*
 * The formula for conics (with x/y swapped as per what I think are 
 * the more natural configuretion) is 
 *
 * The bit of mathematics you need to know is that the surface is defined
 * by the following implicit surface:
 * 
 * (x^2 + y^2) - 2 R z + (K + 1) z^2 == 0
 * 
 * if K == -1, then 
 * 	z = (x^2 + y^2) / (2 * R)
 * else
 * 	z = (R - sqrt((K + 1) (-x^2 - y^2) + R^2 )) / (K + 1)
 *
 * The normal vector is the 2 vector with components:
 * 	-2 * x, -2 * y, 2 R - 2 * (K + 1) z
 *
 */


void
VecNormalize(Vec a)
{
    double l = VecLen(a) ;

    a[0] /= l ; a[1] /= l ; a[2] /= l ;
}

void
VecSub(Vec r, Vec a, Vec b)
{
    r[0] = a[0] - b[0] ;
    r[1] = a[1] - b[1] ;
    r[2] = a[2] - b[2] ;
}

void
Reflect(Vec R, Vec I, Vec N)
{
    double c = -2.0 * VecDot(I, N) ;
    R[0] = c * N[0] + I[0] ;
    R[1] = c * N[1] + I[1] ;
    R[2] = c * N[2] + I[2] ;
    VecNormalize(R);
}

int
main(int argc, char *argv[])
{
    int c, x, y ;
    Vec I, N, R, P, O ;
    double r, fx, fy, gx, t ;

    while ((c = getopt(argc, argv, "d:f:g:k:o:")) != EOF) {
	switch (c) {
	case 'd':
		diameter = atof(optarg) ;
		break ;
	case 'f':
		flen = atof(optarg) ;
		break ;
	case 'g':
		grating = atof(optarg) ;
		break ;
	case 'o':
		offset = atof(optarg) ;
		break ;
	case 'k':
		K = atof(optarg) ;
		break ;
	default:
		abort() ;
	}
    }

    printf("P5\n%d %d\n255\n", RESOLUTION, RESOLUTION) ;

    r = diameter / 2 ;

    O[0] = O[1] = 0.0 ;
    O[2] = 2.0 * flen + offset ;

    for (y=0; y<RESOLUTION; y++) {
	fy = y * diameter / RESOLUTION - diameter / 2.0 ;
	for (x=0; x<RESOLUTION; x++) {
	    fx = x * diameter / RESOLUTION - diameter / 2.0 ;
	    if (fx*fx+fy*fy>r*r) {
		putchar(0) ;
		continue ;
	    }

	    /* construct P */
	    P[0] = fx ;
	    P[1] = fy ;

	    if (fabs(K+1.0) < 1e-5)
		P[2] = (fx*fx+fy*fy)/(4.0 * flen) ;
	    else
	 	P[2] = (2*flen - sqrt((K + 1)*(-fx*fx - fy*fy) + 4*flen*flen)) / (K + 1) ;

	    /* I starts at O and ends at P */
	    VecSub(I, P, O) ;
	    VecNormalize(I) ;

	    N[0] = -2.0 * P[0] ;
	    N[1] = -2.0 * P[1] ;
	    N[2] = 4.0 * flen - 2.0 * (K + 1) * P[2] ;
	    VecNormalize(N) ;

	    Reflect(R, I, N) ;

	    /* compute the intersection of R 
	     * with the plane z = R + offset 
	     */
	    t = (2.0 * flen + offset - P[2]) / R[2] ;
 
	    assert(t > 0.0) ;

	    gx = (P[0] + t * R[0]) * 2.0 * grating - 0.5 ;

	    if (((int)floorf(gx)) & 1) 
		putchar(255) ;
	    else
		putchar(32) ;
	}
    }
}
