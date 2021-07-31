/**
 * @file tool_goal_pose.cpp
 * @brief This defines a cost function for tool goal pose.
 *
 * @author Jorge Nicho
 * @date June 2, 2016
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2016, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <math.h>
#include <stomp_plugins/cost_functions/tool_path_pose.h>
#include <XmlRpcException.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>

PLUGINLIB_EXPORT_CLASS(stomp_moveit::cost_functions::ToolPathPose, stomp_moveit::cost_functions::StompCostFunction);

static const int CARTESIAN_DOF_SIZE = 6;
static const double DEFAULT_POS_TOLERANCE = 0.001;
static const double DEFAULT_ROT_TOLERANCE = 0.01;
static const double POS_MAX_ERROR_RATIO = 10.0;
static const double ROT_MAX_ERROR_RATIO = 10.0;

namespace stomp_moveit
{
  namespace cost_functions
  {

    ToolPathPose::ToolPathPose() : name_("ToolPathPose")
    {
      // TODO Auto-generated constructor stub
    }

    ToolPathPose::~ToolPathPose()
    {
      // TODO Auto-generated destructor stub
    }

    bool ToolPathPose::initialize(moveit::core::RobotModelConstPtr robot_model_ptr,
                                  const std::string &group_name, XmlRpc::XmlRpcValue &config)
    {
      group_name_ = group_name;
      robot_model_ = robot_model_ptr;

      return configure(config);
    }

    bool ToolPathPose::configure(const XmlRpc::XmlRpcValue &config)
    {
      using namespace XmlRpc;

      try
      {
        XmlRpcValue params = config;

        position_cost_weight_ = static_cast<double>(params["position_cost_weight"]);
        // orientation_cost_weight_ = static_cast<double>(params["orientation_cost_weight"]);

        orientation_cost_weight_.x() = static_cast<double>(params["x_orientation_cost_weight_"]);
        orientation_cost_weight_.y() = static_cast<double>(params["y_orientation_cost_weight_"]);
        orientation_cost_weight_.z() = static_cast<double>(params["z_orientation_cost_weight_"]);

        // total weight
        // This isn't used? why is it here
        cost_weight_ = position_cost_weight_ +  orientation_cost_weight_.sum();
      }
      catch (XmlRpc::XmlRpcException &e)
      {
        ROS_ERROR("%s failed to load parameters, %s", getName().c_str(), e.getMessage().c_str());
        return false;
      }

      return true;
    }

    bool ToolPathPose::setMotionPlanRequest(const planning_scene::PlanningSceneConstPtr &planning_scene,
                                            const moveit_msgs::MotionPlanRequest &req,
                                            const stomp_core::StompConfiguration &config,
                                            moveit_msgs::MoveItErrorCodes &error_code)
    {
      using namespace Eigen;
      using namespace moveit::core;

      const JointModelGroup *joint_group = robot_model_->getJointModelGroup(group_name_);
      int num_joints = joint_group->getActiveJointModels().size();
      tool_link_ = joint_group->getLinkModelNames().back();
      state_.reset(new RobotState(robot_model_));
      tool_traj_pose.clear();
      robotStateMsgToRobotState(req.start_state, *state_);

      const std::vector<moveit_msgs::Constraints> &goals = req.goal_constraints;

      const moveit_msgs::TrajectoryConstraints &trajs = req.trajectory_constraints;

      if (trajs.constraints.empty())
      {
        ROS_ERROR("The trajectory constraints were not provided, are you using the wrong thing?");
        error_code.val = error_code.INVALID_MOTION_PLAN;
        return false;
      }

      // storing tool goal pose
      bool found_traj = false;
      for (const auto &g : trajs.constraints)
      {

        if (utils::kinematics::isCartesianConstraints(g))
        {

          // tool cartesian goal data
          state_->updateLinkTransforms();
          Eigen::Affine3d start_tool_pose = state_->getGlobalLinkTransform(tool_link_);
          boost::optional<moveit_msgs::Constraints> cartesian_constraints = utils::kinematics::curateCartesianConstraints(g, start_tool_pose);
          if (cartesian_constraints.is_initialized())
          {
            found_traj = utils::kinematics::decodeCartesianConstraint(robot_model_, cartesian_constraints.get(), tool_goal_pose_,
                                                                      tool_goal_tolerance_, robot_model_->getRootLinkName());
            ROS_DEBUG_STREAM("ToolGoalTolerance cost function will use tolerance: " << tool_goal_tolerance_.transpose());
            tool_traj_pose.push_back(tool_goal_pose_);
          }
          break;
        }

        else
        {
          ROS_ERROR("The trajectory constraints were not Cartesian, Please double check your constraints");
        }
      }

      // setting cartesian error range
      min_twist_error_ = tool_goal_tolerance_;
      max_twist_error_.resize(min_twist_error_.size());
      max_twist_error_.head(3) = min_twist_error_.head(3) * POS_MAX_ERROR_RATIO;
      max_twist_error_.tail(3) = min_twist_error_.tail(3) * ROT_MAX_ERROR_RATIO;

      return true;
    }

    bool ToolPathPose::computeCosts(const Eigen::MatrixXd &parameters,
                                    std::size_t start_timestep,
                                    std::size_t num_timesteps,
                                    int iteration_number,
                                    int rollout_number,
                                    Eigen::VectorXd &costs,
                                    bool &validity)
    {

      using namespace Eigen;
      using namespace utils::kinematics;
      validity = true;

      auto compute_scaled_error = [](const VectorXd &val, VectorXd &min, VectorXd &max) -> VectorXd
      {
        VectorXd capped_val;
        capped_val = (val.array() > max.array()).select(max, val);
        capped_val = (val.array() < min.array()).select(min, val);
        auto range = max - min;
        VectorXd scaled = (capped_val - min).array() / (range.array());
        return scaled;
      };

      Eigen::Affine3d tf;
      Eigen::Vector3d angles_err;
      Eigen::Vector3d pos_err;
      VectorXd scaled_twist_error;


      // preparing cost
      costs.resize(parameters.cols());
      costs.setConstant(0.0);

      for (int idx = 0; idx < tool_traj_pose.size(); idx++)
      {
        joint_pose_ = parameters.col(1);
        state_->setJointGroupPositions(group_name_, joint_pose_);
        state_->updateLinkTransforms();
        tool_pose_ = state_->getGlobalLinkTransform(tool_link_);

        // computing twist error
        tf = tool_traj_pose[idx].inverse() * tool_pose_;
        angles_err = tf.rotation().eulerAngles(2, 1, 0);
        angles_err.reverseInPlace();
        pos_err = tool_goal_pose_.translation() - tool_pose_.translation();

        tool_twist_error_.resize(6);
        tool_twist_error_.head(3) = pos_err.head(3);
        tool_twist_error_.tail(3) = angles_err.tail(3);

        // computing relative error values
        scaled_twist_error = compute_scaled_error(tool_twist_error_, min_twist_error_, max_twist_error_);
        // double pos_error = scaled_twist_error.head(3).cwiseAbs().maxCoeff();
        // double orientation_error = scaled_twist_error.tail(3).cwiseAbs().maxCoeff();

        // Eigen::Vector3d orientation_error = scaled_twist_error(4)*scaled_twist_error(4) * x_orientation_cost_weight_;

        pos_err = scaled_twist_error.head(3) ;
        angles_err = scaled_twist_error.tail(3) ;

        costs(idx) = pos_err.cwiseAbs().maxCoeff() * position_cost_weight_ + angles_err.dot(orientation_cost_weight_);


        // check if valid when twist errors are below the allowed tolerance.
        if (validity) validity = (tool_twist_error_.cwiseAbs().array() <= tool_goal_tolerance_.array()).all();
      }


  
      

      return true;
    }

    void ToolPathPose::done(bool success, int total_iterations, double final_cost, const Eigen::MatrixXd &parameters)
    {
      ROS_DEBUG_STREAM(getName() << " last tool error: " << tool_twist_error_.transpose());
      ROS_DEBUG_STREAM(getName() << " used tool tolerance: " << tool_goal_tolerance_.transpose());
      ROS_DEBUG_STREAM(getName() << " last joint position: " << joint_pose_.transpose());
    }

  } /* namespace cost_functions */
} /* namespace stomp_moveit */
