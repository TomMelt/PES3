        SUBROUTINE PES3(R1,R2,R3,ANS)

        IMPLICIT NONE
        DOUBLE PRECISION, INTENT(IN) :: R1,R2,R3
        DOUBLE PRECISION, INTENT(OUT) :: ANS
        DOUBLE PRECISION SC,VLR,VSR,PEC,SC2,VLR2


c       FOR CALCULATING H2+ PEC
        IF(R1.GE.1000.d0.AND.R2.GE.1000.d0) THEN

        ANS = PEC(R3)

        ELSE IF(R3.LE.R1.AND.R3.LE.R2) THEN
C       THIS IS H2+-PBAR TYPE GEOMETRY

        ANS = (1.D0-SC(R1,R2,R3))*VSR(R1,R2,R3) + 
     1        SC(R1,R2,R3)*VLR(R1,R2,R3)



C       OTHERWISE PN-H GEOMETRY
        ELSE IF(R1.LE.R3.AND.R1.LE.R2) THEN

        ANS = (1.D0-SC2(R1,R2,R3))*VSR(R1,R2,R3) +
     1        SC2(R1,R2,R3)*VLR2(R1,R2,R3)



        ELSE IF(R2.LT.R3.AND.R2.LT.R1) THEN

        ANS = (1.D0-SC2(R1,R2,R3))*VSR(R1,R2,R3) +
     1        SC2(R1,R2,R3)*VLR2(R1,R2,R3)
        

        END IF

        END SUBROUTINE



        DOUBLE PRECISION FUNCTION SC(R1,R2,R3)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,R3,ALF,SL

        ALF = 0.5d0
        SL = 70.0

        SC = 0.5*(1.D0 + TANH(ALF*(R1+R2+R3-SL)))

        RETURN
        END

        DOUBLE PRECISION FUNCTION SC2(R1,R2,R3)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,R3,ALF,SL

        ALF = 1.0d0
        SL = 10.0

        SC2 = 0.5*(1.D0 + TANH(ALF*(R1+R2+R3-SL)))

        RETURN
        END



        DOUBLE PRECISION FUNCTION VSR(R1,R2,R3)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,R3,R(3),S(3),RMIN(3),RMAX(3),P0(626),
     1  V
        INTEGER I,MXPARM,NPARM0

        DATA RMIN/8.0D-002,8.0D-002,0.5d0/,
     1  RMAX/44.d0,44.d0,30.d0/

        MXPARM = 1000
        NPARM0 = 626

        INCLUDE 'NNPARM.DATA125'

        R(1) = R1
        R(2) = R2
        R(3) = R3

        DO I = 1, 3
          S(I) = 2.0D0*(R(I)-RMIN(I))/(RMAX(I)-RMIN(I))-1.0D0
        END DO

        CALL CCH2POT(S,V,P0,MXPARM,NPARM0)

        VSR = V + 1000.D0/abs(R3) - 1000.D0/abs(R1) - 1000.D0/abs(R2)

        RETURN
        END


        DOUBLE PRECISION FUNCTION VLR(R1,R2,R3)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,R3,a,Re,YY,C(11),P(5),
     1  BIGR,GAM,THET,PNR
        DOUBLE PRECISION COSRULE,COSRULE2,LEGE
        INTEGER I

        DATA  A/0.783000000000d+00/
        DATA  RE/ 0.200000000000d+01/
        DATA  C/-0.602633023511d+03,0.298149338751d+00,
     1  0.835868398270d+02,-0.571778035431d+01, 0.120724374515d+02,
     1  0.414345859710d+00, 0.116773037768d+01,0.397150767044d+01,
     1  0.455899232979d+01, 0.196869852245d+01,0.312102488451d+00/
        DATA  P/-0.100000000000d+04, -0.308028258796d+03,
     1  -0.226485956226d+02,0.134530971549d+04,
     1  0.465962203385d+01/

        YY = 1.D0 - EXP(-a*(R3-Re))

        IF(R1.GE.25.0.AND.R2.GE.25.0.AND.R3.LE.25.0) THEN
C       FOR H2+-PBAR GEOMETRIES WHERE PBAR FAR, USE LONG RANGE FORM

       VLR = 0.D0
        DO I = 1, 11
          VLR = VLR + C(I)*(YY**(I-1))
        END DO


        GAM = COSRULE2(R1,R3,R2)
        BIGR = COSRULE(R1,0.5D0*R3,GAM)
        THET = COSRULE2(BIGR,0.5D0*R3,R1)

        VLR = VLR + (P(1)/BIGR) + ((P(2)*R3)/(BIGR**2)) +
     1  (P(3)*(R3**2)*LEGE(2,COS(THET))/(BIGR**2)) +
     1  (P(4)/(BIGR**2)) +
     1  (P(5)*(R3**2)*LEGE(4,COS(THET))/(BIGR**2))


        ELSE 

C       IF ALL PARTICLES FAR USE SEPERATED ATOM LIMIT

        IF(R1.LT.R2) THEN
        PNR = R1
        ELSE IF(R2.LT.R1) THEN
        PNR = R2
        ELSE IF (R1.EQ.R2) THEN
        PNR = R1
        END IF

        VLR = -1000.D0/PNR - 500.D0

        END IF

        RETURN
        END



        DOUBLE PRECISION FUNCTION VLR2(R1,R2,R3)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,R3,P(3),
     1  BIGR,GAM,THET,PNR,SMALLR
        DOUBLE PRECISION COSRULE,COSRULE2,LEGE
        INTEGER I

        DATA  P/0.310049250037E+04,-0.118730742646E+04,
     1  -0.322669101596E+05/

        IF(R1.LT.R2) THEN
        VLR2 = -1000.D0/R1 - 500.d0
        ELSE IF(R1.GE.R2) THEN
        VLR2 = -1000.D0/R2 - 500.D0
        END IF


        IF(R1.LT.R2) THEN
        GAM = COSRULE2(R1,R2,R3)
        BIGR = COSRULE(R2,0.5D0*R1,GAM)
        THET = COSRULE2(BIGR,0.5D0*R1,R2)

         SMALLR = R1

        ELSE IF (R1.GE.R2) THEN
        GAM = COSRULE2(R1,R2,R3)
        BIGR = COSRULE(R1,0.5D0*R2,GAM)
        THET = COSRULE2(BIGR,0.5D0*R2,R1)

        SMALLR = R2

        END IF


        VLR2 = VLR2 + (P(1)*(SMALLR)*LEGE(1,COS(THET))/(BIGR**5)) + 
     1  (P(2)*(SMALLR**2)*LEGE(2,COS(THET))/(BIGR**5)) +
     1  (P(3)*(SMALLR)/(BIGR**6)) 

        IF(R1.GT.8.0.AND.R2.GT.8.0.AND.R3.GT.8.0) THEN

        VLR2 = -1000.D0/SMALLR - 500.D0

        END IF

        RETURN
        END 

        DOUBLE PRECISION FUNCTION PEC(R)

        IMPLICIT NONE
        DOUBLE PRECISION R,a,Re,C(11),YY
        INTEGER I
        DATA a/0.783000000000D+00/
        DATA RE/ 0.200000000000D+01/
        DATA C/-0.602633023511D+03,0.298149338751D+00,
     1  0.835868398270D+02,-0.571778035431D+01, 0.120724374515D+02,
     1  0.414345859710D+00, 0.116773037768D+01,0.397150767044D+01,
     1  0.455899232979D+01, 0.196869852245D+01,0.312102488451D+00/

        YY = 1.D0 - EXP(-a*(R-Re))


       PEC = 0.D0
        DO I = 1, 11
          PEC = PEC + C(I)*(YY**(I-1))
        END DO

        RETURN
        END


         DOUBLE PRECISION FUNCTION LEGE(N,Y)

        DOUBLE PRECISION Y,PN(N+1),PD(N+1)
        INTEGER N

        CALL LPN(N,Y,PN,PD)

        LEGE = PN(N+1)

        RETURN
        END


        SUBROUTINE LPN(N,X,PN,PD)
C
C       ===============================================
C       Purpose: Compute Legendre polynomials Pn(x)
C                and their derivatives Pn'(x)
C       Input :  x --- Argument of Pn(x)
C                n --- Degree of Pn(x) ( n = 0,1,...)
C       Output:  PN(n) --- Pn(x)
C                PD(n) --- Pn'(x)
C       ===============================================
C
        IMPLICIT DOUBLE PRECISION (P,X)
        DIMENSION PN(0:N),PD(0:N)
        PN(0)=1.0D0
        PN(1)=X
        PD(0)=0.0D0
        PD(1)=1.0D0
        P0=1.0D0
        P1=X
        DO 10 K=2,N
           PF=(2.0D0*K-1.0D0)/K*X*P1-(K-1.0D0)/K*P0
           PN(K)=PF
           IF (DABS(X).EQ.1.0D0) THEN
              PD(K)=0.5D0*X**(K+1)*K*(K+1.0D0)
           ELSE
              PD(K)=K*(P1-X*PF)/(1.0D0-X*X)
           ENDIF
           P0=P1
10         P1=PF
        RETURN
        END



        DOUBLE PRECISION FUNCTION COSRULE(R1,R2,THET)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,THET

        COSRULE = SQRT( (R1**2) + (R2**2) -2*R1*R2*COS(THET) )

        RETURN
        END


        DOUBLE PRECISION FUNCTION COSRULE2(R1,R2,R3)

        IMPLICIT NONE
        DOUBLE PRECISION R1,R2,R3

        IF( ((R1**2) + (R2**2) - (R3**2))/(2*R1*R2).LT.(-1.D0)) THEN

        COSRULE2 = ACOS(-1.D0)

        IF( ((R1**2) + (R2**2) - (R3**2))/(2*R1*R2).LT.(-1.01) ) STOP
     1  'PROBLEM IN COSRULE, LT -1.01'

        ELSE

         IF( ((R1**2) + (R2**2) - (R3**2))/(2*R1*R2).GT.(1.D0)) THEN

        COSRULE2 = ACOS(1.D0)

        IF( ((R1**2) + (R2**2) - (R3**2))/(2*R1*R2).GT.(1.01) ) STOP
     1  'PROBLEM IN COSRULE, GT 1.01'

        ELSE

        COSRULE2 = ACOS( ((R1**2) + (R2**2) - (R3**2))/(2*R1*R2))

        END IF
        END IF


        RETURN
        END







