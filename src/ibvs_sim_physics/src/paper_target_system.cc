// The shared target driver integrates the paper path in each physics step.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/transport/Node.hh>

namespace ibvs_sim_physics
{
class PaperTargetSystem : public gz::sim::System,
  public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemReset
{
  public: void Configure(const gz::sim::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &) override
  {
    const gz::sim::Model model(_entity);
    this->targetName = model.Name(_ecm);
    this->link = gz::sim::Link(model.LinkByName(_ecm, "target_link"));
    this->link.EnableVelocityChecks(_ecm);
    this->vehicleName = _sdf->Get<std::string>("vehicle_model");
    this->pattern = _sdf->Get<std::string>("target_pattern");
    this->speed = _sdf->Get<double>("target_speed_m_s");
    this->radiusX = _sdf->Get<double>("radius_x_m");
    this->radiusY = _sdf->Get<double>("radius_y_m");
    this->tilt = _sdf->Get<double>("target_tilt_deg") * std::acos(-1.0) / 180.0;
    if ((this->pattern != "circle" && this->pattern != "figure8") ||
        !std::isfinite(this->speed) || this->speed <= 0.0 ||
        !std::isfinite(this->radiusX) || this->radiusX <= 0.0 ||
        !std::isfinite(this->radiusY) || this->radiusY <= 0.0 ||
        !std::isfinite(this->tilt))
      throw std::invalid_argument("Invalid paper target trajectory");
    // Same arc-length parameterization as PaperTargetTrajectory.
    this->arc.push_back(0.0);
    auto previous = this->Curve(0.0);
    for (int i = 1; i <= kSamples; ++i)
    {
      const auto point = this->Curve(i * this->ThetaStep());
      this->arc.push_back(this->arc.back() + (point - previous).Length());
      previous = point;
    }
    const auto world = gz::sim::World(gz::sim::worldEntity(_ecm)).Name(_ecm).value();
    this->node.Advertise("/world/" + world + "/paper_target/stop",
      &PaperTargetSystem::OnStop, this);
    this->node.Subscribe("/world/" + world + "/model/" + this->targetName +
      "/link/target_link/sensor/target_contact/contact",
      &PaperTargetSystem::OnContact, this);
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;
    if (this->stopped)
    {
      // Release the velocity command so contact can move the lightweight balloon.
      _ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(this->link.Entity());
      return;
    }
    const double dt = std::chrono::duration<double>(_info.dt).count();
    if (dt <= 0.0)
      return;
    if (!this->startTime)
    {
      this->startTime = _info.simTime - _info.dt;
      this->origin = this->link.WorldPose(_ecm)->Pos();
    }
    const double end = std::chrono::duration<double>(
      _info.simTime - *this->startTime).count();
    const auto velocity = (this->Sample(end) - this->Sample(end - dt)) / dt;
    const auto rotation = this->link.WorldPose(_ecm)->Rot();
    this->link.SetLinearVelocity(_ecm, rotation.RotateVectorReverse(velocity));
  }

  public: void Reset(const gz::sim::UpdateInfo &,
      gz::sim::EntityComponentManager &) override
  {
    this->startTime.reset();
    this->stopped = false;
  }

  private: double ThetaStep() const { return 2.0 * std::acos(-1.0) / kSamples; }

  private: gz::math::Vector3d Curve(double _theta) const
  {
    return this->pattern == "circle" ? gz::math::Vector3d(
      this->radiusX * (std::cos(_theta) - 1.0), this->radiusY * std::sin(_theta), 0) :
      gz::math::Vector3d(this->radiusX * std::sin(_theta),
      this->radiusY * std::sin(2.0 * _theta), 0);
  }

  private: gz::math::Vector3d Sample(double _time) const
  {
    const double distance = std::fmod(this->speed * std::max(_time, 0.0),
      this->arc.back());
    const auto upper = std::upper_bound(this->arc.begin(), this->arc.end(), distance);
    const auto index = std::min<std::size_t>(upper - this->arc.begin() - 1,
      kSamples - 1);
    const double fraction = (distance - this->arc[index]) /
      (this->arc[index + 1] - this->arc[index]);
    const auto point = this->Curve((index + fraction) * this->ThetaStep());
    return this->origin + gz::math::Vector3d(point.X() * std::cos(this->tilt),
      point.Y(), point.X() * std::sin(this->tilt));
  }

  private: void OnContact(const gz::msgs::Contacts &_message)
  {
    const auto target = this->targetName + "::target_link::target_collision";
    const auto vehicle = this->vehicleName + "::";
    for (const auto &contact : _message.contact())
      if ((contact.collision1().name() == target &&
           contact.collision2().name().rfind(vehicle, 0) == 0) ||
          (contact.collision2().name() == target &&
           contact.collision1().name().rfind(vehicle, 0) == 0))
        this->stopped = true;
  }

  private: bool OnStop(const gz::msgs::Empty &, gz::msgs::Boolean &_response)
  {
    this->stopped = true;
    _response.set_data(true);
    return true;
  }

  private: static constexpr int kSamples = 8192;
  private: gz::sim::Link link;
  private: gz::transport::Node node;
  private: std::string vehicleName, targetName, pattern;
  private: std::atomic<bool> stopped{false};
  private: double speed{0}, radiusX{0}, radiusY{0}, tilt{0};
  private: gz::math::Vector3d origin;
  private: std::vector<double> arc;
  private: std::optional<std::chrono::steady_clock::duration> startTime;
};
}  // namespace ibvs_sim_physics

GZ_ADD_PLUGIN(ibvs_sim_physics::PaperTargetSystem, gz::sim::System,
  gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate, gz::sim::ISystemReset)
