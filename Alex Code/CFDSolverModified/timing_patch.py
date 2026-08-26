import re, sys

path = "src/time_step_Q2D.f90"

with open(path) as f:
    content = f.read()

def replace_once(content, old, new, label):
    count = content.count(old)
    if count != 1:
        print(f"FAILED: anchor '{label}' found {count} times (expected exactly 1) -- aborting, no changes written.")
        sys.exit(1)
    return content.replace(old, new)

# 1. Declarations
old = "         Integer :: i, j, k, iLo, iHi, jLo, jHi, kLo, kHi\n"
new = old + """
         ! --- TEMPORARY timing instrumentation, remove before committing ---
         Real(kind=8) :: t_launch_temp, t_sync_temp, t_launch_vy, t_sync_vy
         Real(kind=8) :: t_launch_pot, t_sync_pot, t_launch_vx, t_launch_vz
         Real(kind=8) :: t_sync_vx, t_sync_vz, t_launch_prs, t_sync_prs
         Integer, Save :: timing_call_count = 0
"""
content = replace_once(content, old, new, "declarations")

# 2. Temperature launch
old = "         GPU_RHS_T = FDRHP(1:Nx1,1:Ny1,1:Nz1) * GrPr\n         Call solve_eigen_decomp_d(TemperatureHandle, GPU_RHS_T)\n"
new = old + "         Call cpu_time(t_launch_temp)\n"
content = replace_once(content, old, new, "Temperature launch")

# 3. Vy launch
old = "         GPU_RHS_Vy = RHSy(1:Nx1,1:Ny,1:Nz1) / DGr\n         Call solve_eigen_decomp_d(VyHandle, GPU_RHS_Vy)\n"
new = old + "         Call cpu_time(t_launch_vy)\n"
content = replace_once(content, old, new, "Vy launch")

# 4. Potential launch
old = "           Call Get_Potential_Launch\n"
new = old + "           Call cpu_time(t_launch_pot)\n"
content = replace_once(content, old, new, "Potential launch")

# 5. Temperature sync
old = "         Call synch_d(TemperatureHandle, GPU_SOL_T)\n"
new = old + "         Call cpu_time(t_sync_temp)\n"
content = replace_once(content, old, new, "Temperature sync")

# 6. Potential finish (sync)
old = "           Call Get_Potential_Finish\n"
new = old + "           Call cpu_time(t_sync_pot)\n"
content = replace_once(content, old, new, "Potential sync")

# 7. Vx launch
old = "         GPU_RHS_Vx = RHSx(1:Nx,1:Ny1,1:Nz1) / DGr\n         Call solve_eigen_decomp_d(VxHandle, GPU_RHS_Vx)\n"
new = old + "         Call cpu_time(t_launch_vx)\n"
content = replace_once(content, old, new, "Vx launch")

# 8. Vz launch
old = "         GPU_RHS_Vz = RHSz(1:Nx1,1:Ny1,1:Nz) / DGr\n         Call solve_eigen_decomp_d(VzHandle, GPU_RHS_Vz)\n"
new = old + "         Call cpu_time(t_launch_vz)\n"
content = replace_once(content, old, new, "Vz launch")

# 9. Vy sync
old = "         Call synch_d(VyHandle, GPU_SOL_Vy)\n"
new = old + "         Call cpu_time(t_sync_vy)\n"
content = replace_once(content, old, new, "Vy sync")

# 10. Vx sync
old = "         Call synch_d(VxHandle, GPU_SOL_Vx)\n"
new = old + "         Call cpu_time(t_sync_vx)\n"
content = replace_once(content, old, new, "Vx sync")

# 11. Vz sync
old = "         Call synch_d(VzHandle, GPU_SOL_Vz)\n"
new = old + "         Call cpu_time(t_sync_vz)\n"
content = replace_once(content, old, new, "Vz sync")

# 12. Pressure launch + sync
old = "         Call solve_eigen_decomp_d(PressureHandle, GPU_RHS_P)\n         Call synch_d(PressureHandle, Dprs(1:Nx1,1:Ny1,1:Nz1))\n"
new = "         Call solve_eigen_decomp_d(PressureHandle, GPU_RHS_P)\n         Call cpu_time(t_launch_prs)\n         Call synch_d(PressureHandle, Dprs(1:Nx1,1:Ny1,1:Nz1))\n         Call cpu_time(t_sync_prs)\n"
content = replace_once(content, old, new, "Pressure launch+sync")

# 13. Print block before final Return
old = "        Return\n        End\n"
new = """        timing_call_count = timing_call_count + 1
        If (timing_call_count <= 10) Then
            Write(*,*) 'Istp=', Istp
            Write(*,*) '  Temp launch-to-sync gap:', t_sync_temp - t_launch_temp
            Write(*,*) '  Vy   launch-to-sync gap:', t_sync_vy   - t_launch_vy
            Write(*,*) '  Pot  launch-to-sync gap:', t_sync_pot  - t_launch_pot
            Write(*,*) '  Vx   launch-to-sync gap:', t_sync_vx   - t_launch_vx
            Write(*,*) '  Vz   launch-to-sync gap:', t_sync_vz   - t_launch_vz
            Write(*,*) '  Prs  launch-to-sync gap:', t_sync_prs  - t_launch_prs
        End If

        Return
        End
"""
content = replace_once(content, old, new, "final Return block")

with open(path, "w") as f:
    f.write(content)

print("All 13 anchors matched exactly once. Patch applied successfully to", path)
