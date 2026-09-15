/**
 * @file PhasePortrait.h
 * @brief Stateful VTK window for displaying multiple GPU trajectories.
 */

#ifndef PHASE_PORTRAIT_H
#define PHASE_PORTRAIT_H

#include "deviceArrays/headers/Mat.h"

#include <vtkAOSDataArrayTemplate.h>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyLine.h>
#include <vtkProperty.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkSmartPointer.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

/**
 * @brief Owns an interactive VTK window containing multiple trajectories.
 *
 * Each trajectory matrix has three rows and one point per column.
 * draw() adds a curve without removing previous curves.
 * show() starts the blocking interaction loop.
 *
 * All visualization methods must be called on the same GUI thread.
 * CPU visualization storage grows with the curves retained in the window.
 *
 * @tparam Real Coordinate type: float or double.
 */
template<typename Real>
class PhasePortrait {
public:
    /**
     * @brief Creates and connects the window, renderer, and interactor.
     * @param title Window title.
     *
     * Does not start the event loop. Add curves using draw(), then call show().
     */
    explicit PhasePortrait(const char* title = "Phase portrait") {
        renderer_ = vtkSmartPointer<vtkRenderer>::New();
        window_ = vtkSmartPointer<vtkRenderWindow>::New();
        interactor_ = vtkSmartPointer<vtkRenderWindowInteractor>::New();

        renderer_->SetBackground(0.04, 0.05, 0.08);

        window_->SetWindowName(title);
        window_->SetSize(1100, 800);
        window_->AddRenderer(renderer_);

        interactor_->SetRenderWindow(window_);

        vtkNew<vtkInteractorStyleTrackballCamera> style;
        interactor_->SetInteractorStyle(style);

        configureCamera();
    }

    /**
     * @brief Adds a trajectory and renders all retained curves.
     *
     * @param points Three-row GPU matrix with at least two columns.
     * @param handle Stream on which the trajectory is ready or being generated.
     * @param red Red channel in [0, 1].
     * @param green Green channel in [0, 1].
     * @param blue Blue channel in [0, 1].
     *
     * Performs one bulk GPU-to-CPU transfer and waits for completion.
     * The matrix may subsequently be reused; VTK owns its CPU snapshot.
     * Reframes the camera to include every curve.
     *
     * @pre Coordinates are finite; the matrix and handle use the same device.
     * @note Does not start the interaction loop. Call show() after adding curves.
     */
    void draw(
        const Mat<Real>& points,
        Handle& handle,
        double red = 0.15,
        double green = 0.8,
        double blue = 1.0
    ) {

        auto vertices = copyPoints(points, handle);
        auto geometry = createCurve(vertices);
        auto actor = createActor(geometry, red, green, blue);

        renderer_->AddActor(actor);
        renderer_->ResetCamera();
        window_->Render();
    }

    /**
     * @brief Starts mouse interaction with the window.
     *
     * Blocks until the interaction loop exits. Drag to rotate and scroll
     * to zoom. Add initial curves before calling this method.
     */
    void show() {
        interactor_->Initialize();
        window_->Render();
        interactor_->Start();
    }

    /**
     * @brief Removes all curves and renders the empty scene.
     *
     * Releases the renderer's references to the curves and their CPU data.
     * Does not modify any source GPU matrices.
     */
    void clear() {
        renderer_->RemoveAllViewProps();
        window_->Render();
    }

private:
    vtkSmartPointer<vtkRenderer> renderer_;
    vtkSmartPointer<vtkRenderWindow> window_;
    vtkSmartPointer<vtkRenderWindowInteractor> interactor_;

    /**
     * @brief Sets the initial viewing direction with the z axis pointing up.
     */
    void configureCamera() {
        auto* camera = renderer_->GetActiveCamera();
        camera->SetPosition(0.0, -1.0, 0.0);
        camera->SetFocalPoint(0.0, 0.0, 0.0);
        camera->SetViewUp(0.0, 0.0, 1.0);
    }

    /**
     * @brief Transfers the matrix directly into VTK-owned CPU coordinates.
     * @param points Validated trajectory matrix.
     * @param handle Transfer stream.
     * @return Points retaining the transferred coordinate array.
     *
     * A host leading dimension of three removes GPU padding and packs xyz
     * tuples. No intermediate coordinate buffer or per-point copy is used.
     */
    vtkSmartPointer<vtkPoints> copyPoints(
        const Mat<Real>& points,
        Handle& handle
    ) const {
        vtkNew<vtkAOSDataArrayTemplate<Real>> coordinates;
        coordinates->SetNumberOfComponents(3);
        coordinates->SetNumberOfTuples(
            static_cast<vtkIdType>(points._cols)
        );

        points.get(coordinates->GetPointer(0), handle);
        CHECK_CUDA_ERROR(cudaStreamSynchronize(handle));
        coordinates->Modified();

        auto vertices = vtkSmartPointer<vtkPoints>::New();
        vertices->SetData(coordinates);
        return vertices;
    }

    /**
     * @brief Connects successive samples into one continuous curve.
     * @param vertices Trajectory coordinates.
     * @return Geometry retaining coordinates and connectivity.
     *
     * The loop creates connectivity indices only; coordinates are not copied.
     */
    vtkSmartPointer<vtkPolyData> createCurve(vtkPoints* vertices) const {
        vtkNew<vtkPolyLine> curve;
        const vtkIdType count = vertices->GetNumberOfPoints();

        curve->GetPointIds()->SetNumberOfIds(count);
        for (vtkIdType j = 0; j < count; ++j)
            curve->GetPointIds()->SetId(j, j);

        vtkNew<vtkCellArray> lines;
        lines->InsertNextCell(curve);

        auto geometry = vtkSmartPointer<vtkPolyData>::New();
        geometry->SetPoints(vertices);
        geometry->SetLines(lines);
        return geometry;
    }

    /**
     * @brief Creates a colored, unlit trajectory actor.
     * @param geometry Curve geometry.
     * @param red Red channel in [0, 1].
     * @param green Green channel in [0, 1].
     * @param blue Blue channel in [0, 1].
     * @return Actor retaining its mapper and geometry.
     */
    vtkSmartPointer<vtkActor> createActor(
        vtkPolyData* geometry,
        double red,
        double green,
        double blue
    ) const {
        vtkNew<vtkPolyDataMapper> mapper;
        mapper->SetInputData(geometry);

        auto actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(red, green, blue);
        actor->GetProperty()->SetLineWidth(1.5);
        actor->GetProperty()->LightingOff();
        return actor;
    }
};

#endif // PHASE_PORTRAIT_H