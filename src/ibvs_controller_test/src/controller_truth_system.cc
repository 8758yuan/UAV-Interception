// Test-only access to one post-physics snapshot; no pose differentiation.
#include <array>
#include <chrono>
#include <string>

#include <gz/msgs/Utility.hh>
#include <gz/msgs/param_v.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/World.hh>
#include <gz/transport/Node.hh>

namespace ibvs_controller_test
{
class ControllerTruthSystem : public gz::sim::System,
  public gz::sim::ISystemConfigure, public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemPostUpdate
{
  public: void Configure(const gz::sim::Entity &,
      const std::shared_ptr<const sdf::Element> &_sdf,
      gz::sim::EntityComponentManager &_ecm, gz::sim::EventManager &) override
  {
    this->world = gz::sim::World(gz::sim::worldEntity(_ecm));
    this->vehicleName = _sdf->Get<std::string>("vehicle_model");
    this->targetName = _sdf->Get<std::string>("target_model");
    this->publisher = this->node.Advertise<gz::msgs::Param_V>(
      _sdf->Get<std::string>("topic"));
  }

  public: void PreUpdate(const gz::sim::UpdateInfo &,
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

  private: gz::sim::World world;
  private: std::array<gz::sim::Link, 2> links;
  private: gz::transport::Node node;
  private: gz::transport::Node::Publisher publisher;
  private: std::string vehicleName, targetName;
};
}  // namespace ibvs_controller_test

GZ_ADD_PLUGIN(ibvs_controller_test::ControllerTruthSystem, gz::sim::System,
  gz::sim::ISystemConfigure, gz::sim::ISystemPreUpdate,
  gz::sim::ISystemPostUpdate)
