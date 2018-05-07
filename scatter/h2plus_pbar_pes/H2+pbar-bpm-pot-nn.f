      SUBROUTINE CCH2POT(R,V,P0,MXPARM,NPARM0)
C
      IMPLICIT NONE
C
      INTEGER I, J, NNODES, MXPARM, NPARM0
C
      DOUBLE PRECISION V, R(3), P0(MXPARM), W(3), B, C, DN6
      DOUBLE PRECISION EMAX, EMIN

	EMIN = -1700.60
        EMAX = -381.20

C
c      INCLUDE 'emax.h'
C
      IF (MOD(NPARM0-1,5).EQ.0) THEN
        NNODES=(NPARM0-1)/5
C
      ELSE
        WRITE(6,*) 'ERROR IN CCH2POT, NPARM0 = ', NPARM0
        STOP
      ENDIF
C
      V=P0(1)
      DO 500 I = 1, NNODES
        C=P0((I-1)*5+2)
        B=P0((I-1)*5+6)
        DO 400 J = 1, 3
          W(J)=P0((I-1)*5+2+J)
 400    CONTINUE
        V=V+C*DN6(R,W,1,2,3,B)
        V=V+C*DN6(R,W,2,1,3,B)
C
C
 500  CONTINUE
      V= ((V+1)/(2.0D0))*(EMAX-EMIN) + EMIN
C
C
      RETURN
C
      END
C
      DOUBLE PRECISION FUNCTION DN6(X,W,I1,I2,I3,B)
C
      INTEGER I, I1, I2, I3, II(3)
C
      DOUBLE PRECISION X(3), W(3), B
C
      II(1)=I1
      II(2)=I2
      II(3)=I3
C
C
      DN6=1.0D0
      DO 100 I = 1, 3
        DN6=DN6*DEXP(X(I)*W(II(I)))
 100  CONTINUE
      DN6=DN6*DEXP(B)
      DN6=1.0D0/(1.0D0+DN6)
      RETURN
C
      END
C
      DOUBLE PRECISION FUNCTION CCH2ND(R,C,W,B,J)
C
      IMPLICIT NONE
C
      INTEGER IP, J
C
      DOUBLE PRECISION R(3), W(3), B, C, DE6, DN6, RP
      DOUBLE PRECISION EMAX, EMIN
C
      INCLUDE 'emax.h'
C
      IP=MOD(J-1,5)-1
c
c	!!! NEED TO MODIFY FOR IP VALUES
c
      IF (IP.EQ.0) THEN
C
C       Derivative w.r.t. C coefficient
C
        CCH2ND=0.0D0
        CCH2ND=CCH2ND+DN6(R,W,1,2,3,B)
        CCH2ND=CCH2ND+DN6(R,W,2,1,3,B)
      ELSE IF (IP.GE.1.AND.IP.LE.3) THEN
C
C       Derivative w.r.t. weight
C
        CCH2ND=0.0D0
        CCH2ND=CCH2ND-C*DN6(R,W,1,2,3,B)**2
     1 *RP(R,IP,1,2,3)*DE6(R,W,1,2,3,B)
        CCH2ND=CCH2ND-C*DN6(R,W,2,1,3,B)**2
     1 *RP(R,IP,2,1,3)*DE6(R,W,2,1,3,B)
c
C
      ELSE IF (IP.EQ.-1) THEN
C
C       Derivative w.r.t. bias
C
        CCH2ND=0.0D0
        CCH2ND=CCH2ND-C*DN6(R,W,1,2,3,B)**2
     1                 *DE6(R,W,1,2,3,B)
        CCH2ND=CCH2ND-C*DN6(R,W,2,1,3,B)**2
     1                 *DE6(R,W,2,1,3,B)
      ENDIF
C
      CCH2ND=CCH2ND*(EMAX-EMIN)/2.0D0
C
C
      RETURN
C
      END
C  
      DOUBLE PRECISION FUNCTION DE6(X,W,I1,I2,I3,B)
C
      INTEGER I, I1, I2, I3, II(3)
C
      DOUBLE PRECISION X(3), W(3), B
C
      II(1)=I1
      II(2)=I2
      II(3)=I3
      DE6=1.0D0
      DO 100 I = 1, 3
        DE6=DE6*DEXP(X(I)*W(II(I)))
 100  CONTINUE
      DE6=DE6*DEXP(B)
      RETURN
C
      END
C
      DOUBLE PRECISION FUNCTION RP(X,IP,I1,I2,I3)
C
      INTEGER I, IP, I1, I2, I3, I4, I5, I6, II(3)
C
      DOUBLE PRECISION X(3)
C
      II(1)=I1
      II(2)=I2
      II(3)=I3
C
      DO 100 I = 1, 3
        IF (II(I).EQ.IP) THEN
          RP=X(I)
        ENDIF
 100  CONTINUE
C
      RETURN
C
      END
C23456789012345678901234567890123456789012345678901234567890123456789012
