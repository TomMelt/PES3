SUBROUTINE HERMS(ORDER, ALPHA, A, B, X, W)

!*****************************************************************************80
!
!! MAIN is the main program for GEN_HERMITE_RULE.
!
!  Discussion:
!
!    This program computes a generalized Gauss-Hermite quadrature rule
!    and writes it to a file.
!
!    The user specifies:
!    * the ORDER (number of points) in the rule;
!    * ALPHA, the exponent of X;
!    * A, the center point;
!    * B, a scale factor;
!    * FILENAME, the root name of the output files.
!
!  Licensing:
!
!    This code is distributed under the GNU LGPL license.
!
!  Modified:
!
!    22 February 2010
!
!  Author:
!
!    John Burkardt
!
        implicit none

        real(kind=8) a
        real(kind=8) alpha
        integer(kind=4) arg_num
        real(kind=8) b
        real(kind=8) beta
        character(len=255) filename
        integer(kind=4) iarg
        integer(kind=4) iargc
        integer(kind=4) ierror
        integer(kind=4) kind
        integer(kind=4) last
        integer(kind=4) order
        real(kind=8) r(2)
        real(kind=8) r8_huge
        character(len=255) string
        real(kind=8) w(order)
        real(kind=8) x(order)

        arg_num = iargc()

!

        kind = 6
        call cgqf(order, kind, alpha, beta, a, b, x, w)
!

        RETURN
end
