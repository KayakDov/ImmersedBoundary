# DEFINE HERE SOME COMMON COMMANDS, THE COMPILER NAMES, ETC...

# PLAT : use it to add a default suffix to the generated libraries
PLAT    = 
# Library extension, + C and Fortran "-o" option
# may be different under Windows
LIBEXT  = .a
OUTC    = -o 
OUTF    = -o 

# CC : C compiler
CC      = mpiicc
# FC : Fortran 90 compiler
FC      = ifort
# FL : Fortran linker
FL      = ifort


#COMPILER OPTIONS
OPTF    = -O3
OPTC    = -O -I.
OPTL    = -O

#OPTIONS
OPTIONS = -assume buffered_io -mcmodel=large -qopenmp  -I ~/fastspace/MUMPS_5.1.Jun2017_SA/include
