// Test-only access to one post-physics snapshot; no pose differentiation.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gz/msgs/Utility.hh>
#include <gz/msgs/contacts.pb.h>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/param_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/transport/Node.hh>

namespace ibvs_controller_test
{
class ControllerTruthSystem : public gz::sim::System,
  public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemPostUpdate, public gz::sim::ISystemReset
{
  public: void Configure(const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &) override
  {
    this->world = gz::sim::World(gz::sim::worldEntity(_ecm));
    this->vehicleName = _sdf->Get<std::string>("vehicle_model");
    this->targetName = _sdf->Get<std::string>("target_model");
    this->moving = _sdf->Get<bool>("moving_target");
    this->publisher = this->node.Advertise<gz::msgs::Param_V>(
      _sdf->Get<std::string>("topic"));
    this->node.Advertise("/world/" + this->world.Name(_ecm).value() +
      "/controller_test/stop", &ControllerTruthSystem::OnStop, this);
    if (!this->moving)
      return;
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
      throw std::invalid_argument("Invalid controller-test target trajectory");
    // Same 8192-segment arc-length parameterization as PaperTargetTrajectory.
    this->arc.push_back(0.0);
    auto previous = this->Curve(0.0);
    for (int i = 1; i <= kSamples; ++i)
    {
      auto point = this->Curve(i * this->ThetaStep());
      this->arc.push_back(this->arc.back() + (point - previous).Length());
      previous = point;
    }
    const auto topic = "/world/" + this->world.Name(_ecm).value() +
      "/model/" + this->targetName +
      "/link/target_link/sensor/target_contact/contact";
    this->node.Subscribe(topic, &ControllerTruthSystem::OnContact, this);
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &_info,
      gz::sim::EntityComponentManager &_ecm) override
  {
    const std::array<std::string, 2> models{this->vehicleName, this->targetName};
    const std::array<std::string, 2> names{"base_link", "target_link"};
    for (std::size_t i = 0; i < this->links.size(); ++i)
    {
      if (this->links[i].Valid(_ecm))
        continue;
      const auto model = this->world.ModelByName(_ecm, models[i]);
      if (model == gz::sim::kNullEntity)
        return;
      this->links[i] = gz::sim::Link(gz::sim::Model(model).LinkByName(_ecm, names[i]));
      if (!this->links[i].Valid(_ecm))
        return;
      this->links[i].EnableVelocityChecks(_ecm);
    }
    if (!this->moving || _info.paused)
      return;
    if (this->stopped)
    {
      // Removing the command lets the balloon respond freely after contact.
      _ecm.RemoveComponent<gz::sim::components::LinearVelocityCmd>(
        this->links[1].Entity());
      return;
    }
    const double dt = std::chrono::duration<double>(_info.dt).count();
    if (dt <= 0.0)
      return;
    if (!this->startTime)
    {
      this->startTime = _info.simTime - _info.dt;
      this->origin = this->links[1].WorldPose(_ecm)->Pos();
    }
    const double end = std::chrono::duration<double>(
      _info.simTime - *this->startTime).count();
    // Integrate the original path inside each physics step, without teleporting.
    const auto velocity = (this->Sample(end) - this->Sample(end - dt)) / dt;
    const auto rotation = this->links[1].WorldPose(_ecm)->Rot();
    this->links[1].SetLinearVelocity(_ecm, rotation.RotateVectorReverse(velocity));
  }

  public: void PostUpdate(const gz::sim::UpdateInfo &_info,
      const gz::sim::EntityComponentManager &_ecm) override
  {
    if (_info.paused)
      return;
    gz::msgs::Param_V snapshot;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      _info.simTime).count();
    snapshot.mutable_header()->mutable_stamp()->set_sec(ns / 1000000000);
    snapshot.mutable_header()->mutable_stamp()->set_nsec(ns % 1000000000);
    auto &fields = *snapshot.add_param()->mutable_params();
    const std::array<std::string, 2> prefixes{"vehicle", "target"};
    for (std::size_t i = 0; i < this->links.size(); ++i)
    {
      if (!this->links[i].Valid(_ecm))
        return;
      const auto pose = this->links[i].WorldPose(_ecm);
      const auto velocity = this->links[i].WorldLinearVelocity(_ecm);
      const auto angular = this->links[i].WorldAngularVelocity(_ecm);
      if (!pose || !velocity || !angular)
        return;
      auto &p = fields[prefixes[i] + "_pose"];
      p.set_type(gz::msgs::Any::POSE3D);
      gz::msgs::Set(p.mutable_pose3d_value(), *pose);
      auto &v = fields[prefixes[i] + "_velocity"];
      v.set_type(gz::msgs::Any::VECTOR3D);
      gz::msgs::Set(v.mutable_vector3d_value(), *velocity);
      if (i == 0)
      {
        auto &w = fields["vehicle_angular_velocity"];
        w.set_type(gz::msgs::Any::VECTOR3D);
        gz::msgs::Set(w.mutable_vector3d_value(), *angular);
      }
    }
    this->publisher.Publish(snapshot);
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
  private: gz::sim::World world;
  private: std::array<gz::sim::Link, 2> links;
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: std::string vehicleName, targetName, pattern;
  private: bool moving{false};
  private: std::atomic<bool> stopped{false};
  private: double speed{0}, radiusX{0}, radiusY{0}, tilt{0};
  private: gz::math::Vector3d origin;
  private: std::vector<double> arc;
  private: std::optional<std::chrono::steady_clock::duration> startTime;
};
}  // namespace ibvs_controller_test

GZ_ADD_PLUGIN(ibvs_controller_test::ControllerTruthSystem, gz::sim::System,
  gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate,
  gz::sim::ISystemPostUpdate, gz::sim::ISystemReset)
