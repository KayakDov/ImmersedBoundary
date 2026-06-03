include Makefile.inc

OBJS = \
ConvMain_Lid3D.o \
bounds_Lid3D_z.o \
time_step_Lid3D_z.o \
init_Lid3D.o \
pointw.o \
ekinem_3D.o  \
prmesh.o \
MeshStretch.o  \
Solution_time_Lid3D.o \
sublid_3D.o \
outp_Lid3D.o \
BICGf90.o 

MODS = modfv_Lid3D_z.o FVOperators.o EvdProcedures.o IBSetupInterpRegular.o MatrixFormAndOperate.o

SchurSR	:$(MODS) $(OBJS)
	$(FL)  $(MODS) $(OBJS)  -O3 -mcmodel=large -assume buffered_io  -qopenmp -parallel  ~/fastspace/MUMPS_5.1.Jun2017_SA/lib/libdmumps.a ~/fastspace/MUMPS_5.1.Jun2017_SA/lib/libmumps_common.a  ~/fastspace/MUMPS_5.1.Jun2017_SA/lib/libzmumps.a ~/fastspace/dfftpack-master/libdfftpack.a -L ~/fastspace/MUMPS_5.1.Jun2017_SA/PORD/lib -lpord  -L ~/fastspace/MUMPS_5.1.Jun2017_SA/libseq -lmpiseq -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -o $@


.SUFFIXES: .f90 .o
.f90.o:
	$(FC) $(OPTF) $(INCS) $(OPTIONS) -c $*.f90 $(OUTF)$*.o

clean:
	(rm -f  *.o *.mod SchurSR)
