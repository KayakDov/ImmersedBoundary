import sys

path = "src/time_step_Q2D.f90"
with open(path) as f:
    lines = f.readlines()

def find_single(lines, stripped_seq, label):
    n = len(stripped_seq)
    matches = [i for i in range(len(lines) - n + 1)
               if all(lines[i+j].strip() == stripped_seq[j] for j in range(n))]
    if len(matches) != 1:
        print(f"FAILED: anchor '{label}' found {len(matches)} times (expected exactly 1).")
        for m in matches[:5]:
            print(f"   at line {m+1}: {lines[m].strip()!r}")
        sys.exit(1)
    return matches[0]

pos_decl   = find_single(lines, ["Integer :: i, j, k, iLo, iHi, jLo, jHi, kLo, kHi"], "declarations")
pos_temp_l = find_single(lines, ["GPU_RHS_T = FDRHP(1:Nx1,1:Ny1,1:Nz1) * GrPr", "Call solve_eigen_decomp_d(TemperatureHandle, GPU_RHS_T)"], "Temperature launch")
pos_vy_l   = find_single(lines, ["GPU_RHS_Vy = RHSy(1:Nx1,1:Ny,1:Nz1) / DGr", "Call solve_eigen_decomp_d(VyHandle, GPU_RHS_Vy)"], "Vy launch")
pos_pot_l  = find_single(lines, ["Call Get_Potential_Launch"], "Potential launch")
pos_temp_s = find_single(lines, ["Call synch_d(TemperatureHandle, GPU_SOL_T)"], "Temperature sync")
pos_pot_s  = find_single(lines, ["Call Get_Potential_Finish"], "Potential sync")
pos_vx_l   = find_single(lines, ["GPU_RHS_Vx = RHSx(1:Nx,1:Ny1,1:Nz1) / DGr", "Call solve_eigen_decomp_d(VxHandle, GPU_RHS_Vx)"], "Vx launch")
pos_vz_l   = find_single(lines, ["GPU_RHS_Vz = RHSz(1:Nx1,1:Ny1,1:Nz) / DGr", "Call solve_eigen_decomp_d(VzHandle, GPU_RHS_Vz)"], "Vz launch")
pos_vy_s   = find_single(lines, ["Call synch_d(VyHandle, GPU_SOL_Vy)"], "Vy sync")
pos_vx_s   = find_single(lines, ["Call synch_d(VxHandle, GPU_SOL_Vx)"], "Vx sync")
pos_vz_s   = find_single(lines, ["Call synch_d(VzHandle, GPU_SOL_Vz)"], "Vz sync")
pos_prs    = find_single(lines, ["Call solve_eigen_decomp_d(PressureHandle, GPU_RHS_P)", "Call synch_d(PressureHandle, Dprs(1:Nx1,1:Ny1,1:Nz1))"], "Pressure launch+sync")
pos_return = find_single(lines, ["Return", "End"], "final Return block")

print("All 13 anchors matched exactly once.")

decl_text = (
    "\n"
    "         ! --- TEMPORARY timing instrumentation, remove before committing ---\n"
    "         Real(kind=8) :: t_launch_temp, t_sync_temp, t_launch_vy, t_sync_vy\n"
    "         Real(kind=8) :: t_launch_pot, t_sync_pot, t_launch_vx, t_launch_vz\n"
    "         Real(kind=8) :: t_sync_vx, t_sync_vz, t_launch_prs, t_sync_prs\n"
    "         Integer, Save :: timing_call_count = 0\n"
)

print_block_text = (
    "        timing_call_count = timing_call_count + 1\n"
    "        If (timing_call_count <= 10) Then\n"
    "            Write(*,*) 'Istp=', Istp\n"
    "            Write(*,*) '  Temp launch-to-sync gap:', t_sync_temp - t_launch_temp\n"
    "            Write(*,*) '  Vy   launch-to-sync gap:', t_sync_vy   - t_launch_vy\n"
    "            Write(*,*) '  Pot  launch-to-sync gap:', t_sync_pot  - t_launch_pot\n"
    "            Write(*,*) '  Vx   launch-to-sync gap:', t_sync_vx   - t_launch_vx\n"
    "            Write(*,*) '  Vz   launch-to-sync gap:', t_sync_vz   - t_launch_vz\n"
    "            Write(*,*) '  Prs  launch-to-sync gap:', t_sync_prs  - t_launch_prs\n"
    "        End If\n"
    "\n"
)

insertions = [
    (pos_decl+1,   decl_text),
    (pos_temp_l+2, "         Call cpu_time(t_launch_temp)\n"),
    (pos_vy_l+2,   "         Call cpu_time(t_launch_vy)\n"),
    (pos_pot_l+1,  "           Call cpu_time(t_launch_pot)\n"),
    (pos_temp_s+1, "         Call cpu_time(t_sync_temp)\n"),
    (pos_pot_s+1,  "           Call cpu_time(t_sync_pot)\n"),
    (pos_vx_l+2,   "         Call cpu_time(t_launch_vx)\n"),
    (pos_vz_l+2,   "         Call cpu_time(t_launch_vz)\n"),
    (pos_vy_s+1,   "         Call cpu_time(t_sync_vy)\n"),
    (pos_vx_s+1,   "         Call cpu_time(t_sync_vx)\n"),
    (pos_vz_s+1,   "         Call cpu_time(t_sync_vz)\n"),
    (pos_prs+1,    "         Call cpu_time(t_launch_prs)\n"),
    (pos_prs+2,    "         Call cpu_time(t_sync_prs)\n"),
    (pos_return,   print_block_text),
]

insertions.sort(key=lambda t: t[0], reverse=True)
for idx, text in insertions:
    lines.insert(idx, text)

with open(path, "w") as f:
    f.writelines(lines)

print("Patch applied successfully to", path)
