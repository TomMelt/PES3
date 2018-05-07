objects = diatomic.o herm.o lagtest.o pot.o
COMP = ifort
OPT = -mkl

diatom: $(objects)
	$(COMP) $(OPT) -o diatom $(objects)

%.o: %.f
	$(COMP) -c $<

%.o: %.f90
	$(COMP) -c $<

clean:
	rm *.o

cleanall:
	rm *.o diatom
